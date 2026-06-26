# LARE-KD: Lesion-Aware Regional Enhancement and Adaptive Temperature Scaling Self-Knowledge Distillation

LARE-KD is a PyTorch implementation for medical image classification. It extends
temperature scaling self-knowledge distillation with two practical modules:

- **ARE: Attention-guided Regional Enhancement** uses the model's layer4 feature
  response to locate high-response regions and applies local CLAHE enhancement to
  those regions.
- **UATS: Uncertainty-guided Adaptive Temperature Scaling** estimates teacher
  uncertainty and assigns sample-wise non-target temperature multipliers for
  self-knowledge distillation.
- **Stability extensions** add an optional EMA teacher, effective-number
  class-balanced CE, label smoothing, and flip-based test-time augmentation
  (TTA). These are the recommended switches when trying to improve the ACC/AUC
  numbers reported for TSS-KD.

By default the code can still run in the original self-distillation setting:
ResNet18 is used as the backbone, the teacher view is computed with
`torch.no_grad()`, and only the student view is optimized. When
`--use_ema_teacher True` is enabled, the no-gradient teacher branch is replaced
by an exponential moving average copy of the student for a more stable target.

## Method Overview

Training uses two augmented views of each image:

1. **Teacher view**: weakly augmented image, forwarded without gradient.
2. **Student view**: augmented image, optionally enhanced by ARE after warm-up.
3. **Loss**: cross entropy plus temperature-scaled KD loss.
4. **Optional EMA teacher**: the KD target and ARE attention are generated from
   a smoothed copy of the online student.
5. **Optional TTA evaluation**: validation/test logits are averaged over
   deterministic flip views.

The training objective is:

```text
loss = CE(student_logits, label) + kd_weight * tau^2 * KD(student, teacher)
```

`kd_weight` is `--lamda` by default and can be linearly warmed up with
`--kd_weight_rampup`.

## Supported Datasets

MedMNIST datasets are downloaded through `medmnist`:

- `oct`
- `path`
- `derma`
- `organ_a`
- `organ_c`
- `organ_s`

External datasets use the `torchvision.datasets.ImageFolder` format:

- `isic_m`
- `isic_k`
- `cbis`

Expected directory layout:

```text
data/
  isic_m/
    train/class0/*.jpg
    train/class1/*.jpg
    val/class0/*.jpg
    val/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
  isic_k/
    train/class0/*.jpg
    train/class1/*.jpg
    val/class0/*.jpg
    val/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
  cbis/
    train/class0/*.jpg
    train/class1/*.jpg
    val/class0/*.jpg
    val/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
```

You can also pass `--train_dir`, `--val_dir`, and `--test_dir` to directly
specify ImageFolder paths. MedMNIST uses its official `train`, `val`, and `test`
splits.

## Installation

The target environment is Python 3.8 with PyTorch 1.12.1 and CUDA 11.6.

```bash
pip install -r requirements.txt
```

ARE depends on OpenCV CLAHE. If OpenCV is missing, install it with:

```bash
pip install opencv-python
```

## Training Commands

Original TSS-KD style baseline:

```bash
python main.py --dataset derma --use_are False --use_adaptive_temp False
```

ARE only:

```bash
python main.py --dataset derma \
  --use_are True \
  --use_adaptive_temp False \
  --are_warmup 20 \
  --are_prob 0.5 \
  --clahe_clip_limit 1.5
```

UATS only:

```bash
python main.py --dataset derma \
  --use_are False \
  --use_adaptive_temp True \
  --temp_warmup 20 \
  --lambda_min 0.8 \
  --lambda_max 1.2 \
  --entropy_momentum 0.95
```

Conservative full LARE-KD:

```bash
python main.py --dataset derma \
  --use_are True \
  --use_adaptive_temp True \
  --are_warmup 20 \
  --temp_warmup 20 \
  --are_prob 0.5 \
  --clahe_clip_limit 1.5 \
  --lambda_min 0.8 \
  --lambda_max 1.2 \
  --entropy_momentum 0.95 \
  --kd_weight_rampup 10
```

Performance-oriented LARE-KD for chasing the table metrics:

```bash
python main.py --dataset derma \
  --use_are True \
  --use_adaptive_temp True \
  --use_ema_teacher True \
  --eval_ema True \
  --use_class_balanced_ce True \
  --use_balanced_sampler True \
  --logit_adjust_tau 1.0 \
  --label_smoothing 0.05 \
  --tta_views 4 \
  --are_warmup 20 \
  --temp_warmup 20 \
  --are_prob 0.5 \
  --clahe_clip_limit 1.5 \
  --lambda_strategy confidence_inverse \
  --lambda_min 0.8 \
  --lambda_max 1.2 \
  --entropy_momentum 0.95 \
  --kd_weight_rampup 10
```

Binary validation-tuned ACC run:

```bash
python main.py --dataset isic_m --data_root ./data \
  --use_are True \
  --use_adaptive_temp True \
  --use_ema_teacher True \
  --eval_ema True \
  --use_class_balanced_ce True \
  --use_balanced_sampler True \
  --logit_adjust_tau 1.0 \
  --label_smoothing 0.05 \
  --tta_views 4 \
  --use_val_threshold True
```

`--use_val_threshold True` searches the best positive-class probability
threshold on the validation split and applies that fixed threshold to test ACC
and F1. This is the recommended fair setting for binary tasks. The older
`--search_acc_threshold True` option searches the threshold on the evaluated
split itself; keep it for debugging or validation analysis, not final test
reporting.

External ImageFolder example:

```bash
python main.py --dataset isic_m --data_root ./data \
  --use_are True --use_adaptive_temp True \
  --use_ema_teacher True --use_class_balanced_ce True \
  --use_balanced_sampler True --tta_views 4
```

## Important Options

- `--use_are`: enable attention-guided local CLAHE.
- `--are_warmup`: epochs before ARE starts.
- `--are_prob`: probability of applying ARE to each eligible sample.
- `--teacher_conf_thresh`: minimum teacher confidence required for ARE.
- `--clahe_clip_limit`: CLAHE clip limit. Lower values are more conservative.
- `--disable_rga_when_are`: disable random gamma crop when ARE is enabled.
- `--use_adaptive_temp`: enable sample-wise adaptive temperature.
- `--lambda_strategy`: `entropy_linear` or `confidence_inverse`.
- `--lambda_min`, `--lambda_max`: adaptive lambda range.
- `--kd_conf_thresh`: minimum teacher confidence required for KD contribution.
- `--kd_weight_rampup`: linearly warm up the KD loss weight.
- `--use_ema_teacher`: use an EMA copy as the teacher branch.
- `--ema_decay`: EMA update decay. The default is `0.999`.
- `--eval_ema`: evaluate and save EMA weights when EMA teacher is enabled.
- `--use_class_balanced_ce`: use effective-number class weights for CE.
- `--cb_beta`: beta value for class-balanced CE.
- `--use_balanced_sampler`: use inverse-frequency weighted sampling.
- `--logit_adjust_tau`: add class-prior log probabilities to training CE logits.
  This helps long-tailed datasets when set around `0.5` to `1.0`.
- `--label_smoothing`: CE label smoothing.
- `--tta_views`: deterministic test-time augmentation views. Use `4` for
  original, horizontal flip, vertical flip, and both flips.
- `--val_dir`: optional ImageFolder validation directory.
- `--use_val_threshold`: tune a binary threshold on validation and apply it to
  test ACC/F1.
- `--fixed_acc_threshold`: manually set a binary threshold for ACC/F1.
- `--search_acc_threshold`: optionally search the best binary ACC threshold on
  the evaluation split.
- `--save`: output directory. If omitted, a timestamped directory is created
  under `./result/`.

## Outputs

Each run writes these files under the save directory:

- `epoch_metrics.txt`: one tab-separated row per epoch.
- `summary.txt`: run config, last epoch metrics, best AUC, best ACC, and paths.
- `best_auc_model.pth` and `best_auc_checkpoint.pth`.
- `best_acc_model.pth` and `best_acc_checkpoint.pth`.
- `best_model.pth` and `best_checkpoint.pth`: aliases for the best-AUC model.

Training logs include loss, CE loss, KD loss, accuracy, mean lambda, mean
entropy, mean teacher confidence, ARE application ratio, and KD active ratio.
`summary.txt` also records class counts, class weights, EMA/TTA settings, and
all LARE-KD control flags used for the run. For binary tasks it also records the
probability threshold used for ACC/F1.

## Practical Notes

- The first serious run should use EMA teacher, class-balanced CE, balanced
  sampling, logit adjustment, and TTA. They directly target the most common
  failure modes in DermaMNIST, ISIC, and CBIS-DDSM: unstable teacher targets,
  class imbalance, and fixed-threshold ACC sensitivity.
- If LARE-KD performs worse than the baseline, first try conservative settings:
  `--lambda_min 0.8 --lambda_max 1.2 --are_prob 0.5 --clahe_clip_limit 1.5`.
- Use `epoch_metrics.txt` to check whether the issue is early instability,
  overfitting, or a specific module hurting performance.
- For weak teacher predictions, try `--kd_conf_thresh 0.5` or a longer
  `--temp_warmup`.
- For noisy local enhancement, lower `--are_prob`, increase `--are_warmup`, or
  use a smaller `--clahe_clip_limit`.
- Recommended ablation order:
  1. baseline TSS-KD;
  2. baseline + EMA teacher;
  3. baseline + EMA + class-balanced CE;
  4. baseline + EMA + class-balanced CE + balanced sampler + logit adjustment;
  5. full LARE-KD + EMA + class-balanced CE + balanced sampler + TTA.

## Quick Verification

```bash
python -m py_compile main.py adaptive_temp.py attention_enhance.py datapreprocess.py training_utils.py
```
