# LARE-KD

**Lesion-Aware Regional Enhancement and Adaptive Temperature Scaling Self-Knowledge Distillation**

LARE-KD is a PyTorch project for medical image classification. It is built on
temperature scaling self-knowledge distillation and extends it with lesion-aware
regional enhancement, uncertainty-guided adaptive temperature scaling, and a
lightweight class prototype regularization module.

The code keeps the original TSS-KD training style available: a ResNet18 student
receives two augmented views of the same image, the teacher branch is evaluated
with `torch.no_grad()`, and only the student branch is optimized. All new modules
are controlled by command-line switches and are disabled by default.

## Method

The current implementation contains four main parts:

- **Self-KD backbone**: ResNet18 produces logits and a 128-dimensional embedding.
  The teacher and student share the same network by default.
- **ARE**: Attention-guided Regional Enhancement uses the teacher feature map to
  find high-response regions and applies local CLAHE to the selected region.
- **UATS**: Uncertainty-guided Adaptive Temperature Scaling estimates teacher
  uncertainty and creates a sample-wise non-target temperature multiplier.
- **CPR**: Class Prototype Regularization maintains EMA class prototypes in the
  embedding space and adds a prototype classification loss.

The training loss is:

```text
loss = CE(student_logits, label)
     + kd_weight * tau^2 * KD(student_logits, teacher_logits)
     + proto_weight * CPR(student_embedding, label)
```

`CPR` is optional. It reuses the student embedding and does not add an extra
backbone forward.

## Repository Structure

```text
LARE-KD/
  main.py                 # training, validation, testing, logging, checkpoints
  Net.py                  # ResNet18 model with optional feature/embedding output
  adaptive_temp.py        # UATS and adaptive TSKD loss
  attention_enhance.py    # attention map, attention boxes, local CLAHE
  datapreprocess.py       # two-view transform wrapper and ImageFolder helpers
  training_utils.py       # EMA, class balance, TTA, thresholds, CPR utilities
  utils.py                # original TSS-KD utilities and gamma augmentation
  requirements.txt        # Python dependencies
  tests/test_lare_kd.py   # utility tests
```

## Environment

Recommended environment:

- Python 3.8
- PyTorch 1.12.1
- CUDA 11.6

Install dependencies:

```bash
pip install -r requirements.txt
```

If OpenCV is missing, install it manually:

```bash
pip install opencv-python
```

## Supported Datasets

MedMNIST datasets are downloaded through the `medmnist` package:

- `oct`
- `path`
- `derma`
- `organ_a`
- `organ_c`
- `organ_s`

External datasets use `torchvision.datasets.ImageFolder`:

- `isic_m`
- `isic_k`
- `cbis`

Expected external dataset layout:

```text
data/
  isic_m/
    train/class0/*.jpg
    train/class1/*.jpg
    val/class0/*.jpg
    val/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
```

The same layout is used for `isic_k` and `cbis`. You can also pass explicit
paths with `--train_dir`, `--val_dir`, and `--test_dir`.

Dataset defaults:

| Dataset type | Image size | Default batch | Default epochs | ResNet first conv |
| --- | ---: | ---: | ---: | --- |
| MedMNIST | 32 | 128 | 100 | `first_conv=False` |
| ISIC / CBIS ImageFolder | 224 | 64 | 200 | `first_conv=True` |

## Quick Start

Original TSS-KD style baseline:

```bash
python main.py --dataset derma --use_are False --use_adaptive_temp False
```

CPR-only run, recommended as the first low-risk improvement test:

```bash
python main.py --dataset derma \
  --use_are False \
  --use_adaptive_temp False \
  --use_proto_loss True \
  --proto_weight 0.1 \
  --proto_temp 0.2 \
  --proto_warmup 5
```

Conservative full LARE-KD:

```bash
python main.py --dataset derma \
  --use_are True \
  --use_adaptive_temp True \
  --use_proto_loss True \
  --are_warmup 30 \
  --temp_warmup 30 \
  --are_prob 0.25 \
  --clahe_clip_limit 1.2 \
  --lambda_min 0.9 \
  --lambda_max 1.1 \
  --lamda 0.5 \
  --kd_weight_rampup 30 \
  --proto_weight 0.1
```

External ImageFolder binary task:

```bash
python main.py --dataset isic_m --data_root ./data \
  --use_proto_loss True \
  --use_ema_teacher True \
  --eval_ema True \
  --use_val_threshold True \
  --tta_views 4
```

## Module Commands

ARE only:

```bash
python main.py --dataset derma \
  --use_are True \
  --use_adaptive_temp False \
  --are_warmup 30 \
  --are_prob 0.25 \
  --clahe_clip_limit 1.2
```

UATS only:

```bash
python main.py --dataset derma \
  --use_are False \
  --use_adaptive_temp True \
  --temp_warmup 30 \
  --lambda_strategy confidence_inverse \
  --lambda_min 0.9 \
  --lambda_max 1.1 \
  --entropy_momentum 0.95 \
  --kd_weight_rampup 30
```

EMA teacher and TTA:

```bash
python main.py --dataset derma \
  --use_ema_teacher True \
  --eval_ema True \
  --tta_views 4
```

Class-imbalance options:

```bash
python main.py --dataset derma \
  --use_class_balanced_ce True \
  --label_smoothing 0.05
```

For strongly imbalanced external data, test these options carefully:

```bash
python main.py --dataset isic_m --data_root ./data \
  --use_class_balanced_ce True \
  --use_balanced_sampler True \
  --logit_adjust_tau 0.5
```

Avoid enabling too many imbalance corrections before you have a baseline. Start
with class-balanced CE, then add sampler or logit adjustment only if needed.

## Important Arguments

General:

- `--dataset`: one of `oct`, `path`, `derma`, `organ_a`, `organ_c`, `organ_s`,
  `isic_m`, `isic_k`, `cbis`.
- `--data_root`: root directory for ImageFolder datasets.
- `--train_dir`, `--val_dir`, `--test_dir`: explicit ImageFolder paths.
- `--batchSz`, `--nEpochs`, `--lr`, `--opt`: basic training settings.
- `--save`: output directory. If omitted, the code creates a timestamped folder
  under `./result/`.

ARE:

- `--use_are`: enable attention-guided local CLAHE.
- `--are_warmup`: epochs before ARE starts.
- `--are_prob`: probability of applying ARE to eligible samples.
- `--topk_ratio`: high-attention region ratio.
- `--min_box_ratio`: minimum attention box size.
- `--clahe_clip_limit`: CLAHE strength. Lower is more conservative.
- `--teacher_conf_thresh`: minimum teacher confidence for ARE.
- `--disable_rga_when_are`: disable random gamma crop when ARE is enabled.

UATS / KD:

- `--use_adaptive_temp`: enable sample-wise adaptive temperature.
- `--lambda_strategy`: `entropy_linear` or `confidence_inverse`.
- `--lambda_min`, `--lambda_max`: adaptive lambda range.
- `--entropy_momentum`: EMA momentum for entropy memory.
- `--temp_warmup`: epochs before adaptive temperature starts.
- `--fixed_lambda`: fixed lambda before warm-up or when UATS is disabled.
- `--kd_conf_thresh`: minimum teacher confidence for KD contribution.
- `--kd_weight_rampup`: linearly warm up KD loss weight.
- `--temp`: KD temperature.
- `--lamda`: KD loss weight.

CPR:

- `--use_proto_loss`: enable class prototype regularization.
- `--proto_weight`: CPR loss weight. Start with `0.1`.
- `--proto_temp`: prototype classification temperature. Default is `0.2`.
- `--proto_momentum`: EMA momentum for class prototypes.
- `--proto_warmup`: epochs before CPR contributes to the loss.
- `--proto_min_count`: minimum same-class samples in a batch before updating a
  prototype.

Stability and evaluation:

- `--use_ema_teacher`: use an EMA model as the teacher branch.
- `--ema_decay`: EMA decay.
- `--eval_ema`: evaluate and save the EMA model when available.
- `--tta_views`: deterministic flip TTA views.
- `--use_class_balanced_ce`: use effective-number class-balanced CE.
- `--cb_beta`: beta for class-balanced CE.
- `--use_balanced_sampler`: use inverse-frequency weighted sampling.
- `--logit_adjust_tau`: class-prior logit adjustment strength.
- `--label_smoothing`: label smoothing for CE.
- `--use_val_threshold`: tune binary ACC/F1 threshold on validation split.
- `--fixed_acc_threshold`: manually set binary threshold.
- `--search_acc_threshold`: search threshold on the evaluated split. Use this
  for debugging, not final test reporting.

## Outputs

Each run writes files to the save directory:

- `epoch_metrics.txt`: tab-separated metrics for every epoch.
- `summary.txt`: run config, last epoch metrics, best AUC, best ACC, and paths.
- `best_auc_model.pth`
- `best_auc_checkpoint.pth`
- `best_acc_model.pth`
- `best_acc_checkpoint.pth`
- `best_model.pth`
- `best_checkpoint.pth`

`best_model.pth` and `best_checkpoint.pth` are aliases for the best-AUC model.
Checkpoints include model weights, optimizer state, arguments, train/test
metrics, and CPR prototype state when available.

Training logs report:

- total loss
- CE loss
- KD loss
- CPR loss
- train ACC
- mean lambda
- mean teacher entropy
- mean teacher confidence
- ARE active ratio
- KD active ratio
- CPR active classes

## Recommended Experiment Order

Use this order when the current metrics are low:

1. Baseline TSS-KD.
2. Baseline + CPR.
3. Baseline + EMA teacher.
4. Baseline + CPR + EMA teacher.
5. Add class-balanced CE if the dataset is imbalanced.
6. Add UATS with a narrow lambda range, such as `0.9` to `1.1`.
7. Add ARE last, with conservative CLAHE settings.

This order helps identify whether the performance drop comes from image
enhancement, adaptive temperature, class imbalance handling, or teacher
instability.

## Practical Tips

- If AUC and ACC are both low, first disable ARE and UATS, then test CPR and EMA.
- If AUC is acceptable but ACC is low on binary datasets, use
  `--use_val_threshold True`.
- If ARE hurts performance, increase `--are_warmup`, lower `--are_prob`, or
  reduce `--clahe_clip_limit`.
- If UATS hurts performance, use `--lambda_min 0.9 --lambda_max 1.1` and a longer
  `--temp_warmup`.
- If KD is noisy early in training, use `--kd_weight_rampup 20` or `30`.
- If the dataset is highly imbalanced, do not stack every imbalance trick at
  once. Test CE weights, sampler, and logit adjustment separately.

## Verification

Static syntax check:

```bash
python -m py_compile main.py Net.py adaptive_temp.py attention_enhance.py datapreprocess.py training_utils.py tests/test_lare_kd.py
```

Unit tests:

```bash
python -m unittest tests.test_lare_kd
```

A quick one-epoch smoke test:

```bash
python main.py --dataset derma --nEpochs 1 --batchSz 8 --use_proto_loss True --save ./tmp_proto_test
```

## Notes

This project is research code. The best configuration may differ across
MedMNIST, ISIC, and CBIS-DDSM. Always compare new modules against a clean
baseline and use `epoch_metrics.txt` plus `summary.txt` to diagnose the effect of
each switch.
