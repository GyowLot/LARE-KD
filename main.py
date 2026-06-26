#  在CS-KD基础上改进

from medmnist import OCTMNIST, PathMNIST, DermaMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import sys
import math
import numpy as np
from datetime import datetime
from sklearn import metrics
from Net import *
from datapreprocess import *
from utils import *
from adaptive_temp import (
    adaptive_tskd_loss,
    compute_entropy,
    compute_lambda_from_confidence,
    compute_lambda_from_entropy,
    init_entropy_memory,
    update_entropy_memory,
)
from attention_enhance import apply_local_clahe_tensor, generate_attention_map, get_attention_boxes
from training_utils import (
    apply_logit_adjustment,
    build_class_balanced_weights,
    build_sample_weights,
    create_ema_model,
    extract_class_counts,
    extract_labels,
    find_best_binary_threshold,
    tta_forward,
    update_ema_model,
)
import setproctitle

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


MEDMNIST_REGISTRY = {
    "oct": OCTMNIST,
    "path": PathMNIST,
    "derma": DermaMNIST,
    "organ_a": OrganAMNIST,
    "organ_c": OrganCMNIST,
    "organ_s": OrganSMNIST,
}

DATASET_PROFILES = {
    "oct": {"source": "medmnist", "class_num": 4, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "path": {"source": "medmnist", "class_num": 9, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "derma": {"source": "medmnist", "class_num": 7, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "organ_a": {"source": "medmnist", "class_num": 11, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "organ_c": {"source": "medmnist", "class_num": 11, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "organ_s": {"source": "medmnist", "class_num": 11, "image_size": 32, "batch_size": 128, "epochs": 100, "first_conv": False},
    "isic_m": {"source": "imagefolder", "class_num": 2, "image_size": 224, "batch_size": 64, "epochs": 200, "first_conv": True},
    "isic_k": {"source": "imagefolder", "class_num": 2, "image_size": 224, "batch_size": 64, "epochs": 200, "first_conv": True},
    "cbis": {"source": "imagefolder", "class_num": 2, "image_size": 224, "batch_size": 64, "epochs": 200, "first_conv": True},
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean value expected")


def denormalize_tensor(images, mean, std):
    return images * std + mean


def normalize_tensor(images, mean, std):
    return (images - mean) / std


def get_run_mode(args):
    """Return the run mode name from ARE/UATS switches."""
    if args.use_are and args.use_adaptive_temp:
        return "larekd"
    if args.use_are:
        return "are"
    if args.use_adaptive_temp:
        return "uats"
    return "tsskd"


def build_save_dir(args):
    """Build a dynamic save directory when --save is not provided."""
    if args.save:
        return args.save
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "{}_{}_lr{}_bs{}_ep{}_temp{}_{}".format(
        args.dataset,
        get_run_mode(args),
        args.lr,
        args.batchSz,
        args.nEpochs,
        args.temp,
        timestamp,
    )
    return os.path.join("./result", run_name)


def format_flag(value):
    """Return a compact ON/OFF string for terminal logs."""
    return "ON " if value else "OFF"


def print_run_config(args, num_params):
    """Print the main run configuration in a readable block."""
    print("\n" + "=" * 72)
    print(" LARE-KD Training")
    print("=" * 72)
    print(" Dataset        : {:<12} Source      : {}".format(args.dataset, args.dataset_source))
    print(" Classes        : {:<12} Image size  : {}".format(args.class_num, args.image_size))
    print(" Batch size     : {:<12} Epochs      : {}".format(args.batchSz, args.nEpochs))
    print(" Optimizer      : {:<12} LR          : {}".format(args.opt, args.lr))
    print(" Label smooth   : {:<12} CB loss     : {}".format(args.label_smoothing, format_flag(args.use_class_balanced_ce).strip()))
    print(" Balanced samp. : {:<12} Logit adj.  : {}".format(format_flag(args.use_balanced_sampler).strip(), args.logit_adjust_tau))
    print(" Temperature    : {:<12} Fixed lambda: {}".format(args.temp, args.fixed_lambda))
    print(" ARE            : {:<12} Warmup      : {}".format(format_flag(args.use_are), args.are_warmup))
    print(" ARE prob       : {:<12} CLAHE clip  : {}".format(args.are_prob, args.clahe_clip_limit))
    print(" Teacher conf   : {:<12} KD conf     : {}".format(args.teacher_conf_thresh, args.kd_conf_thresh))
    print(" Adaptive temp  : {:<12} Warmup      : {}".format(format_flag(args.use_adaptive_temp), args.temp_warmup))
    print(" Lambda strategy: {:<12} KD rampup   : {}".format(args.lambda_strategy, args.kd_weight_rampup))
    print(" EMA teacher    : {:<12} Decay       : {}".format(format_flag(args.use_ema_teacher).strip(), args.ema_decay))
    print(" Eval EMA       : {:<12} TTA views   : {}".format(format_flag(args.eval_ema).strip(), args.tta_views))
    print(" ACC threshold  : {}".format(format_flag(args.search_acc_threshold).strip()))
    print(" Save dir       : {}".format(args.save))
    if args.dataset_source == "imagefolder":
        print(" Train dir      : {}".format(args.train_dir))
        print(" Test dir       : {}".format(args.test_dir))
    print(" Parameters     : {:,}".format(num_params))
    print("=" * 72)


def print_epoch_header(epoch, total_epochs):
    """Print a clear epoch separator."""
    print("\n" + "-" * 72)
    print(" Epoch {:03d}/{}".format(epoch, total_epochs))
    print("-" * 72)


def print_batch_metrics(partial_epoch, metrics):
    """Print one compact batch-progress row."""
    print(
        "[train {:7.2f}] "
        "loss {loss:8.4f} | ce {ce_loss:8.4f} | kd {kd_loss:8.4f} | "
        "acc {acc:6.2%} | lambda {mean_lambda:6.3f} | entropy {mean_entropy:6.3f} | "
        "conf {mean_teacher_conf:6.3f} | ARE {are} {are_active_ratio:5.1%} | "
        "KD {kd_active_ratio:5.1%} | UATS {uats}".format(
            partial_epoch,
            loss=metrics["loss"],
            ce_loss=metrics["ce_loss"],
            kd_loss=metrics["kd_loss"],
            acc=metrics["acc"],
            mean_lambda=metrics["mean_lambda"],
            mean_entropy=metrics["mean_entropy"],
            mean_teacher_conf=metrics["mean_teacher_conf"],
            are_active_ratio=metrics["are_active_ratio"],
            kd_active_ratio=metrics["kd_active_ratio"],
            are=format_flag(metrics["are_enabled"]).strip(),
            uats=format_flag(metrics["adaptive_temp_enabled"]).strip(),
        )
    )


def print_train_summary(epoch, metrics):
    """Print the train summary for one epoch."""
    print(
        "[train summary] epoch {:03d} | loss {:8.4f} | ce {:8.4f} | kd {:8.4f} | "
        "acc {:6.2%} | lambda {:6.3f} | entropy {:6.3f} | conf {:6.3f} | "
        "ARE {} {:5.1%} | KD {:5.1%} | UATS {}".format(
            epoch,
            metrics["loss"],
            metrics["ce_loss"],
            metrics["kd_loss"],
            metrics["acc"],
            metrics["mean_lambda"],
            metrics["mean_entropy"],
            metrics["mean_teacher_conf"],
            format_flag(metrics["are_enabled"]).strip(),
            metrics["are_active_ratio"],
            metrics["kd_active_ratio"],
            format_flag(metrics["adaptive_temp_enabled"]).strip(),
        )
    )


def print_test_summary(epoch, metrics):
    """Print the test summary for one epoch."""
    print(
        "[test  summary] epoch {:03d} | loss {:8.4f} | acc {:6.2%} | "
        "auc {:6.4f} | f1 {:6.4f} | thr {:5.3f}".format(
            epoch,
            metrics["loss"],
            metrics["acc"],
            metrics["auc"],
            metrics["f1"],
            metrics["threshold"],
        )
    )


def print_best_message(metric_name, epoch, value):
    """Print a concise best-checkpoint notification."""
    print("[best {:>3}] epoch {:03d} | {} {:.4f} | checkpoint saved".format(metric_name, epoch, metric_name, value))


def write_epoch_metrics(log_path, metrics_dict):
    """Append one epoch of train/test metrics to a tab-separated text file."""
    fieldnames = [
        "epoch", "train_loss", "train_ce_loss", "train_kd_loss", "train_acc",
        "mean_lambda", "mean_entropy", "mean_teacher_conf",
        "are_active_ratio", "kd_active_ratio",
        "are_enabled", "adaptive_temp_enabled",
        "test_loss", "test_acc", "test_auc", "test_f1", "test_threshold",
        "best_auc", "best_acc", "is_best_auc", "is_best_acc",
    ]
    needs_header = not os.path.exists(log_path)
    with open(log_path, "a") as f:
        if needs_header:
            f.write("\t".join(fieldnames) + "\n")
        f.write("\t".join(str(metrics_dict[name]) for name in fieldnames) + "\n")


def save_best_model(args, epoch, net, optimizer, train_metrics, test_metrics,
                    metric_name, best_metric, model_path, checkpoint_path):
    """Save the best model files and return a text-summary dictionary."""
    torch.save(net, model_path)
    torch.save(
        {
            "epoch": epoch,
            "metric_name": metric_name,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "args": vars(args),
        },
        checkpoint_path,
    )
    return {
        "metric_name": metric_name,
        "epoch": epoch,
        "best_metric": best_metric,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_threshold": test_metrics["threshold"],
        "train_loss": train_metrics["loss"],
        "train_ce_loss": train_metrics["ce_loss"],
        "train_kd_loss": train_metrics["kd_loss"],
        "train_acc": train_metrics["acc"],
        "mean_lambda": train_metrics["mean_lambda"],
        "mean_entropy": train_metrics["mean_entropy"],
        "mean_teacher_conf": train_metrics["mean_teacher_conf"],
        "are_active_ratio": train_metrics["are_active_ratio"],
        "kd_active_ratio": train_metrics["kd_active_ratio"],
        "are_enabled": train_metrics["are_enabled"],
        "adaptive_temp_enabled": train_metrics["adaptive_temp_enabled"],
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
    }


def write_summary(summary_path, args, last_record, best_auc_info, best_acc_info):
    """Write one centralized summary file for the current run."""
    with open(summary_path, "w") as f:
        f.write("LARE-KD run summary\n")
        f.write("=" * 72 + "\n")
        f.write("dataset: {}\n".format(args.dataset))
        f.write("mode: {}\n".format(get_run_mode(args)))
        f.write("save_dir: {}\n".format(args.save))
        f.write("class_num: {}\n".format(args.class_num))
        f.write("image_size: {}\n".format(args.image_size))
        f.write("batch_size: {}\n".format(args.batchSz))
        f.write("epochs: {}\n".format(args.nEpochs))
        f.write("lr: {}\n".format(args.lr))
        f.write("label_smoothing: {}\n".format(args.label_smoothing))
        f.write("use_class_balanced_ce: {}\n".format(args.use_class_balanced_ce))
        f.write("cb_beta: {}\n".format(args.cb_beta))
        f.write("use_balanced_sampler: {}\n".format(args.use_balanced_sampler))
        f.write("logit_adjust_tau: {}\n".format(args.logit_adjust_tau))
        f.write("class_counts: {}\n".format(args.class_counts))
        f.write("class_weights: {}\n".format(args.class_weights))
        f.write("temp: {}\n".format(args.temp))
        f.write("fixed_lambda: {}\n".format(args.fixed_lambda))
        f.write("use_are: {}\n".format(args.use_are))
        f.write("are_prob: {}\n".format(args.are_prob))
        f.write("clahe_clip_limit: {}\n".format(args.clahe_clip_limit))
        f.write("teacher_conf_thresh: {}\n".format(args.teacher_conf_thresh))
        f.write("disable_rga_when_are: {}\n".format(args.disable_rga_when_are))
        f.write("use_adaptive_temp: {}\n".format(args.use_adaptive_temp))
        f.write("lambda_strategy: {}\n".format(args.lambda_strategy))
        f.write("kd_conf_thresh: {}\n".format(args.kd_conf_thresh))
        f.write("kd_weight_rampup: {}\n".format(args.kd_weight_rampup))
        f.write("use_ema_teacher: {}\n".format(args.use_ema_teacher))
        f.write("ema_decay: {}\n".format(args.ema_decay))
        f.write("eval_ema: {}\n".format(args.eval_ema))
        f.write("tta_views: {}\n".format(args.tta_views))
        f.write("search_acc_threshold: {}\n".format(args.search_acc_threshold))
        f.write("data_root: {}\n".format(args.data_root))
        f.write("train_dir: {}\n".format(args.train_dir))
        f.write("test_dir: {}\n".format(args.test_dir))

        f.write("\nLast epoch metrics\n")
        f.write("-" * 72 + "\n")
        for key in [
            "epoch", "train_loss", "train_ce_loss", "train_kd_loss", "train_acc",
            "mean_lambda", "mean_entropy", "mean_teacher_conf",
            "are_active_ratio", "kd_active_ratio",
            "test_loss", "test_acc", "test_auc",
            "test_f1", "test_threshold", "best_auc", "best_acc",
        ]:
            f.write("{}: {}\n".format(key, last_record[key]))

        for title, info in [("Best AUC", best_auc_info), ("Best ACC", best_acc_info)]:
            f.write("\n{}\n".format(title))
            f.write("-" * 72 + "\n")
            if info is None:
                f.write("not available\n")
                continue
            for key in [
                "epoch", "best_metric", "test_loss", "test_acc", "test_auc", "test_f1",
                "test_threshold",
                "train_loss", "train_ce_loss", "train_kd_loss", "train_acc",
                "mean_lambda", "mean_entropy", "mean_teacher_conf",
                "are_active_ratio", "kd_active_ratio",
                "model_path", "checkpoint_path",
            ]:
                f.write("{}: {}\n".format(key, info[key]))
 
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
    
class DKDLoss(nn.Module):
    def __init__(self, temp_factor, t_error):
        super(DKDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.t_error = t_error
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input_1, input_2, target):
        
        tea_error = self.t_error * self.temp_factor * torch.ones_like(input_1)
        indexs = torch.arange(len(input_1))
        tea_error[indexs, target] = self.temp_factor
        
        log_p = torch.log_softmax(input_1/tea_error, dim=1)
        q = torch.softmax(input_2/tea_error, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input_1.size(0)
        
        
#         log_p = torch.log_softmax(input/self.temp_factor, dim=1)
#         q = torch.softmax(target/self.temp_factor, dim=1)
#         loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='oct',
                        choices=tuple(DATASET_PROFILES.keys()),
                        help='dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='root directory for ImageFolder datasets')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='optional ImageFolder train directory override')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='optional ImageFolder test directory override')
    parser.add_argument('--batchSz', type=int, default=None,
                        help='batch size; defaults depend on dataset')
    parser.add_argument('--nEpochs', type=int, default=None,
                        help='number of epochs; defaults depend on dataset')
    parser.add_argument('--class_num', type=int, default=None,
                        help='number of classes; defaults to the selected MedMNIST dataset')
    parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
    parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
    parser.add_argument('--alpha_T',default=0.8 ,type=float, help='alpha_T')
    parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--t_error',default=0.1 ,type=float, help='alpha_T')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--save', default=None,
                        help='save directory; defaults to an auto-generated run directory')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='label smoothing for the training CE loss')
    parser.add_argument('--use_class_balanced_ce', type=str2bool, default=False,
                        help='use effective-number class-balanced CE weights')
    parser.add_argument('--cb_beta', default=0.9999, type=float,
                        help='beta for effective-number class-balanced CE')
    parser.add_argument('--use_balanced_sampler', type=str2bool, default=False,
                        help='use inverse-frequency weighted sampling for the train loader')
    parser.add_argument('--logit_adjust_tau', default=0.0, type=float,
                        help='class-prior logit adjustment strength for the training CE loss')
    parser.add_argument('--use_are', type=str2bool, default=False,
                        help='enable Attention-guided Regional Enhancement')
    parser.add_argument('--are_warmup', type=int, default=10,
                        help='epochs before enabling ARE')
    parser.add_argument('--are_prob', type=float, default=0.5,
                        help='probability of applying ARE to each eligible sample')
    parser.add_argument('--topk_ratio', type=float, default=0.2,
                        help='top-k attention ratio for ARE boxes')
    parser.add_argument('--min_box_ratio', type=float, default=0.25,
                        help='minimum ARE box size ratio')
    parser.add_argument('--clahe_clip_limit', type=float, default=1.5,
                        help='CLAHE clip limit used by ARE')
    parser.add_argument('--teacher_conf_thresh', type=float, default=0.0,
                        help='minimum teacher confidence for applying ARE')
    parser.add_argument('--disable_rga_when_are', type=str2bool, default=True,
                        help='disable random gamma crop whenever ARE is enabled')
    parser.add_argument('--use_adaptive_temp', type=str2bool, default=False,
                        help='enable uncertainty-guided adaptive temperature')
    parser.add_argument('--lambda_strategy', type=str, default='entropy_linear',
                        choices=('entropy_linear', 'confidence_inverse'),
                        help='strategy for computing adaptive lambda')
    parser.add_argument('--lambda_min', type=float, default=0.5,
                        help='minimum adaptive lambda')
    parser.add_argument('--lambda_max', type=float, default=2.0,
                        help='maximum adaptive lambda')
    parser.add_argument('--entropy_momentum', type=float, default=0.9,
                        help='EMA momentum for entropy memory')
    parser.add_argument('--temp_warmup', type=int, default=10,
                        help='epochs before enabling adaptive temperature')
    parser.add_argument('--fixed_lambda', type=float, default=None,
                        help='fixed lambda used before warm-up or when adaptive temperature is disabled')
    parser.add_argument('--entropy_init', type=str, default='uniform',
                        choices=('uniform', 'zero'),
                        help='entropy memory initialization')
    parser.add_argument('--kd_conf_thresh', type=float, default=0.0,
                        help='minimum teacher confidence for KD; lower-confidence samples get zero KD weight')
    parser.add_argument('--kd_weight_rampup', type=int, default=0,
                        help='linearly ramp KD loss weight over this many epochs; 0 disables ramp-up')
    parser.add_argument('--use_ema_teacher', type=str2bool, default=False,
                        help='use an EMA copy as the no-grad teacher branch')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA teacher decay')
    parser.add_argument('--eval_ema', type=str2bool, default=True,
                        help='evaluate and save the EMA model when EMA teacher is enabled')
    parser.add_argument('--tta_views', type=int, default=1,
                        help='test-time augmentation views: 1 original, 2 +hflip, 3 +vflip, 4 +hvflip')
    parser.add_argument('--search_acc_threshold', type=str2bool, default=False,
                        help='search the best binary ACC threshold on the evaluation split')

    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_known_args()[0]
    profile = DATASET_PROFILES[args.dataset]
    args.batchSz = profile["batch_size"] if args.batchSz is None else args.batchSz
    args.nEpochs = profile["epochs"] if args.nEpochs is None else args.nEpochs
    args.image_size = profile["image_size"]
    args.first_conv = profile["first_conv"]
    args.dataset_source = profile["source"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.fixed_lambda = args.t_error if args.fixed_lambda is None else args.fixed_lambda
    args.class_num = profile["class_num"] if args.class_num is None else args.class_num
    args.save = build_save_dir(args)
    if not 0.0 <= args.are_prob <= 1.0:
        raise ValueError("--are_prob must be in [0, 1]")
    if args.clahe_clip_limit <= 0:
        raise ValueError("--clahe_clip_limit must be > 0")
    if not 0.0 <= args.teacher_conf_thresh <= 1.0:
        raise ValueError("--teacher_conf_thresh must be in [0, 1]")
    if not 0.0 <= args.kd_conf_thresh <= 1.0:
        raise ValueError("--kd_conf_thresh must be in [0, 1]")
    if args.kd_weight_rampup < 0:
        raise ValueError("--kd_weight_rampup must be >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0, 1)")
    if not 0.0 <= args.cb_beta < 1.0:
        raise ValueError("--cb_beta must be in [0, 1)")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema_decay must be in [0, 1)")
    if args.tta_views < 1:
        raise ValueError("--tta_views must be >= 1")
    if args.logit_adjust_tau < 0:
        raise ValueError("--logit_adjust_tau must be >= 0")

    setproctitle.setproctitle(args.save)
    criterion_2 = None

    os.makedirs(args.save, exist_ok=True)

    normTransform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = [
        transforms.RandomResizedCrop(args.image_size,scale=(0.8, 1.0)),
    ]
    if args.dataset_source == "imagefolder":
        train_transforms.append(transforms.RandomRotation(10))
    if (not args.use_are) or (not args.disable_rga_when_are):
        train_transforms.append(gammaCrop(0.5,2))
    train_transforms.extend([
        transforms.ColorJitter(contrast=0.5, saturation=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform
    ])
    trainTransform = transforms.Compose(train_transforms)
    testTransform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        normTransform
    ])
    
    if args.dataset_source == "medmnist":
        dataset_cls = MEDMNIST_REGISTRY[args.dataset]
        BCdata_trian_base = dataset_cls(split='train', transform=ContrastiveLearningViewGenerator(trainTransform,n_views=2),download=True)
        BCdata_test = dataset_cls(split='test', transform=ContrastiveLearningViewGenerator(testTransform,n_views=1),download=True)
    else:
        train_dir = args.train_dir or os.path.join(args.data_root, args.dataset, "train")
        test_dir = args.test_dir or os.path.join(args.data_root, args.dataset, "test")
        args.train_dir = train_dir
        args.test_dir = test_dir
        BCdata_trian_base = build_imagefolder_dataset(train_dir, trainTransform, n_views=2)
        BCdata_test = build_imagefolder_dataset(test_dir, testTransform, n_views=1)
        if len(BCdata_trian_base.classes) != args.class_num:
            raise ValueError(
                "Dataset {} expects {} classes, but ImageFolder found {} classes: {}".format(
                    args.dataset, args.class_num, len(BCdata_trian_base.classes), BCdata_trian_base.classes
                )
            )
        if BCdata_test.classes != BCdata_trian_base.classes:
            raise ValueError(
                "Train/test class folders do not match. train={}, test={}".format(
                    BCdata_trian_base.classes, BCdata_test.classes
                )
            )
    BCdata_trian = IndexedDataset(BCdata_trian_base)
    class_counts = extract_class_counts(BCdata_trian_base, args.class_num)
    args.class_counts = [int(x) for x in class_counts.cpu().tolist()]
    class_weights = None
    args.class_weights = None
    if args.use_class_balanced_ce:
        class_weights = build_class_balanced_weights(
            class_counts,
            beta=args.cb_beta,
            device=device,
        )
        args.class_weights = [float(x) for x in class_weights.detach().cpu().tolist()]
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    test_criterion = nn.CrossEntropyLoss()
    train_sampler = None
    train_shuffle = True
    if args.use_balanced_sampler:
        train_labels = extract_labels(BCdata_trian_base)
        sample_weights = build_sample_weights(train_labels, args.class_num)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    
    trainLoader = DataLoader(
        BCdata_trian,
        batch_size=args.batchSz,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        num_workers=4,
    )
    testLoader = DataLoader(BCdata_test, batch_size=args.batchSz, shuffle=False, drop_last=False, num_workers=4)
    
    net = Student_resnet18(first_conv=args.first_conv, class_num=args.class_num)
    net = net.to(device) 
    ema_net = create_ema_model(net) if args.use_ema_teacher else None
    args.device = device
    args.norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    args.norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    entropy_memory = init_entropy_memory(
        len(BCdata_trian),
        args.class_num,
        init=args.entropy_init,
        device=device,
    )

    num_params = sum([p.data.nelement() for p in net.parameters()])
    print_run_config(args, num_params)

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr,betas=(0.9, 0.999), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size= 10,
        iter_max= args.nEpochs,
        power= 0.9,
    )

    epoch_log_path = os.path.join(args.save, "epoch_metrics.txt")
    summary_path = os.path.join(args.save, "summary.txt")
    best_model_path = os.path.join(args.save, "best_model.pth")
    best_checkpoint_path = os.path.join(args.save, "best_checkpoint.pth")
    best_auc_model_path = os.path.join(args.save, "best_auc_model.pth")
    best_auc_checkpoint_path = os.path.join(args.save, "best_auc_checkpoint.pth")
    best_acc_model_path = os.path.join(args.save, "best_acc_model.pth")
    best_acc_checkpoint_path = os.path.join(args.save, "best_acc_checkpoint.pth")
    best_auc = -float("inf")
    best_acc = -float("inf")
    best_auc_info = None
    best_acc_info = None
    
    for epoch in range(1, args.nEpochs + 1):
        train_metrics = train(args, epoch, net, ema_net, trainLoader, optimizer, criterion,criterion_2,scheduler, entropy_memory)
        eval_net = ema_net if (args.use_ema_teacher and args.eval_ema) else net
        test_metrics = test(args, epoch, eval_net, testLoader,optimizer, test_criterion)
        torch.save(eval_net, os.path.join(args.save, str(epoch)+'.pth'))
        is_best_auc = test_metrics["auc"] > best_auc
        is_best_acc = test_metrics["acc"] > best_acc
        if is_best_auc:
            best_auc = test_metrics["auc"]
        if is_best_acc:
            best_acc = test_metrics["acc"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_acc": train_metrics["acc"],
            "mean_lambda": train_metrics["mean_lambda"],
            "mean_entropy": train_metrics["mean_entropy"],
            "mean_teacher_conf": train_metrics["mean_teacher_conf"],
            "are_active_ratio": train_metrics["are_active_ratio"],
            "kd_active_ratio": train_metrics["kd_active_ratio"],
            "are_enabled": train_metrics["are_enabled"],
            "adaptive_temp_enabled": train_metrics["adaptive_temp_enabled"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_auc": test_metrics["auc"],
            "test_f1": test_metrics["f1"],
            "test_threshold": test_metrics["threshold"],
            "best_auc": best_auc,
            "best_acc": best_acc,
            "is_best_auc": is_best_auc,
            "is_best_acc": is_best_acc,
        }
        write_epoch_metrics(epoch_log_path, epoch_record)

        if is_best_auc:
            best_auc_info = save_best_model(
                args, epoch, eval_net, optimizer, train_metrics, test_metrics,
                metric_name="AUC", best_metric=best_auc,
                model_path=best_auc_model_path,
                checkpoint_path=best_auc_checkpoint_path,
            )
            save_best_model(
                args, epoch, eval_net, optimizer, train_metrics, test_metrics,
                metric_name="AUC", best_metric=best_auc,
                model_path=best_model_path,
                checkpoint_path=best_checkpoint_path,
            )
            print_best_message("AUC", epoch, best_auc)

        if is_best_acc:
            best_acc_info = save_best_model(
                args, epoch, eval_net, optimizer, train_metrics, test_metrics,
                metric_name="ACC", best_metric=best_acc,
                model_path=best_acc_model_path,
                checkpoint_path=best_acc_checkpoint_path,
            )
            print_best_message("ACC", epoch, best_acc)
        write_summary(summary_path, args, epoch_record, best_auc_info, best_acc_info)


    

    
def train(args, epoch, net, ema_net, trainLoader, optimizer, criterion, criterion_2, scheduler, entropy_memory):
    net.train()                                       # 设置网络为训练模式
    
 
    teacher_net = ema_net if args.use_ema_teacher else net
    if teacher_net is not net:
        teacher_net.eval()

    print_epoch_header(epoch, args.nEpochs)
    nProcessed = 0
    total_loss = 0
    total_correct = 0
    total_ce_loss = 0
    total_kd_loss = 0
    total_lambda = 0
    total_entropy = 0
    total_teacher_conf = 0
    total_are_active = 0
    total_kd_active = 0
    seen_samples = 0
    nTrain = len(trainLoader.dataset)
    are_enabled = args.use_are and epoch >= args.are_warmup
    adaptive_temp_enabled = args.use_adaptive_temp and epoch >= args.temp_warmup
#     print(nTrain)
    for batch_idx, (pos_1, target, indices) in enumerate(trainLoader):

        # cross entropy loss 
        images_1 = pos_1[0].to(args.device)
        images_2 = pos_1[1].to(args.device)
        correct = 0
        target = target.to(args.device).long().view(-1)
        indices = indices.to(args.device).long().view(-1)
        
#         images_2,_ = gammamix(images_2,0.5,2)
        # gammamix
        with torch.no_grad():
            # another view
            output_last, teacher_feat = teacher_net(images_1, return_feature=True)
            teacher_prob = F.softmax(output_last.detach(), dim=1)
            teacher_conf = teacher_prob.max(dim=1)[0]
            if are_enabled:
                attention_map = generate_attention_map(teacher_feat, out_size=images_2.shape[-2:])
                boxes = get_attention_boxes(
                    attention_map,
                    topk_ratio=args.topk_ratio,
                    min_box_ratio=args.min_box_ratio,
                )
                are_mask = torch.rand(images_2.size(0), device=images_2.device) < args.are_prob
                if args.teacher_conf_thresh > 0:
                    are_mask = are_mask & (teacher_conf >= args.teacher_conf_thresh)
                if are_mask.any():
                    masked_boxes = [
                        box if bool(are_mask[i].item()) else (0, 0, 0, 0)
                        for i, box in enumerate(boxes)
                    ]
                    images_2_unit = denormalize_tensor(images_2, args.norm_mean, args.norm_std)
                    images_2_unit = apply_local_clahe_tensor(
                        images_2_unit,
                        masked_boxes,
                        clip_limit=args.clahe_clip_limit,
                    )
                    images_2 = normalize_tensor(images_2_unit, args.norm_mean, args.norm_std)
                are_active_count = int(are_mask.sum().detach().cpu().item())
            else:
                are_active_count = 0
            
        output = net(images_2)
        
        ce_logits = apply_logit_adjustment(output, args.class_counts, tau=args.logit_adjust_tau)
        loss_1 = criterion(ce_logits,target)
#         loss_2 = criterion_2(output, output_last.detach())
        entropy = compute_entropy(teacher_prob)
        if adaptive_temp_enabled:
            update_entropy_memory(
                entropy_memory,
                indices,
                entropy,
                momentum=args.entropy_momentum,
            )
            if args.lambda_strategy == "confidence_inverse":
                lambda_i = compute_lambda_from_confidence(
                    teacher_conf,
                    args.class_num,
                    lambda_min=args.lambda_min,
                    lambda_max=args.lambda_max,
                )
            else:
                lambda_i = compute_lambda_from_entropy(
                    entropy_memory[indices],
                    args.class_num,
                    lambda_min=args.lambda_min,
                    lambda_max=args.lambda_max,
                )
        else:
            lambda_i = torch.full(
                (output.size(0),),
                args.fixed_lambda,
                device=output.device,
                dtype=output.dtype,
            )

        if args.kd_conf_thresh > 0:
            kd_sample_weight = (teacher_conf >= args.kd_conf_thresh).to(output.dtype)
        else:
            kd_sample_weight = torch.ones(output.size(0), device=output.device, dtype=output.dtype)
        kd_active_count = int(kd_sample_weight.detach().sum().cpu().item())
        loss_2 = adaptive_tskd_loss(
            output,
            output_last.detach(),
            target,
            args.temp,
            lambda_i,
            sample_weight=kd_sample_weight,
        )
        if args.kd_weight_rampup > 0:
            kd_weight = args.lamda * min(float(epoch) / float(args.kd_weight_rampup), 1.0)
        else:
            kd_weight = args.lamda
        loss = loss_1 + kd_weight * (args.temp ** 2) * loss_2
        

        total_loss = loss.item()+total_loss
        total_ce_loss += loss_1.item()
        total_kd_loss += loss_2.item()
        total_lambda += float(lambda_i.detach().mean().cpu()) * output.size(0)
        total_entropy += float(entropy.detach().mean().cpu()) * output.size(0)
        total_teacher_conf += float(teacher_conf.detach().mean().cpu()) * output.size(0)
        total_are_active += are_active_count
        total_kd_active += kd_active_count
        seen_samples += output.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_net is not None:
            update_ema_model(ema_net, net, decay=args.ema_decay)
        
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().int().cpu().numpy()
        total_correct = total_correct+correct

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if((batch_idx%100)==0):
            print_batch_metrics(
                partialEpoch,
                {
                    "loss": loss.item(),
                    "ce_loss": loss_1.item(),
                    "kd_loss": loss_2.item(),
                    "acc": correct/output.size(0),
                    "mean_lambda": float(lambda_i.detach().mean().cpu()),
                    "mean_entropy": float(entropy.detach().mean().cpu()),
                    "mean_teacher_conf": float(teacher_conf.detach().mean().cpu()),
                    "are_active_ratio": are_active_count / output.size(0),
                    "kd_active_ratio": kd_active_count / output.size(0),
                    "are_enabled": are_enabled,
                    "adaptive_temp_enabled": adaptive_temp_enabled,
                },
            )
    
    scheduler.step()
    
    denom_batches = max(len(trainLoader), 1)
    denom_samples = max(seen_samples, 1)
    train_metrics = {
        "loss": total_loss/denom_batches,
        "ce_loss": total_ce_loss/denom_batches,
        "kd_loss": total_kd_loss/denom_batches,
        "acc": total_correct/denom_samples,
        "mean_lambda": total_lambda/denom_samples,
        "mean_entropy": total_entropy/denom_samples,
        "mean_teacher_conf": total_teacher_conf/denom_samples,
        "are_active_ratio": total_are_active/denom_samples,
        "kd_active_ratio": total_kd_active/denom_samples,
        "are_enabled": are_enabled,
        "adaptive_temp_enabled": adaptive_temp_enabled,
    }
    print_train_summary(epoch, train_metrics)
    return train_metrics




def test(args, epoch, net, testLoader,optimizer, criterion):
    net.eval()
    total_loss = 0
    total_correct = 0
    conMatrix_pre = []
    conMatrix_tar = []
    AUC_data = []
    AUC_target = []
    sensitivity = 0
    specificity = 0
    nTrain = len(testLoader.dataset)
    

    with torch.no_grad():
        for pos_1, target in testLoader:
#         for pos_1, target, _ in testLoader:
#         for pos_1, _ , target in testLoader:
            # cross entropy loss
            images = pos_1.to(args.device)
            target = target.to(args.device).long().view(-1)
            
            output = tta_forward(net, images, tta_views=args.tta_views)
            loss = criterion(output,target)
            total_loss = loss.item() * images.size(0) + total_loss
            b,_ = output.size()
            output = F.softmax(output,dim=1)
#             print(prediction)
            for i in range(len(output)):
                if args.class_num == 2:
                    conMatrix_pre.append(float(output[i][1].cpu().detach()))   #  for 2 class
                else:
                    conMatrix_pre.append(output[i].cpu().detach().numpy())
                conMatrix_tar.append(int(target[i].cpu().detach().numpy()))
                
            prediction = torch.argmax(output, 1)      
            total_correct += (prediction == target).sum().int().cpu().numpy()
            
#     print(conMatrix_pre)
    if args.class_num == 2:
        test_AUC = metrics.roc_auc_score(np.array(conMatrix_tar), np.array(conMatrix_pre))
        threshold = 0.5
        test_acc = total_correct/nTrain
        if args.search_acc_threshold:
            threshold_tensor, acc_tensor = find_best_binary_threshold(
                torch.tensor(conMatrix_pre),
                torch.tensor(conMatrix_tar),
            )
            threshold = float(threshold_tensor)
            test_acc = float(acc_tensor)
        pred_for_f1 = (np.array(conMatrix_pre) >= threshold).astype(np.int64)
    else:
        test_AUC = metrics.roc_auc_score(np.array(conMatrix_tar), np.array(conMatrix_pre), multi_class='ovo')
        threshold = -1.0
        test_acc = total_correct/nTrain
        pred_for_f1 = np.argmax(np.array(conMatrix_pre), axis=1)
    test_f1 = metrics.f1_score(np.array(conMatrix_tar), pred_for_f1, average='macro')

    test_metrics = {
        "loss": total_loss/nTrain,
        "acc": test_acc,
        "auc": test_AUC,
        "f1": test_f1,
        "threshold": threshold,
    }
    print_test_summary(epoch, test_metrics)
    return test_metrics


if __name__=='__main__':
    main()
