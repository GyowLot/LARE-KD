# LARE-KD: Lesion-Aware Regional Enhancement and Adaptive Temperature Scaling Self-Knowledge Distillation

LARE-KD is a PyTorch implementation for medical image classification. It extends
temperature scaling self-knowledge distillation with two practical modules:

- **ARE: Attention-guided Regional Enhancement** uses the model's layer4 feature
  response to locate high-response regions and applies local CLAHE enhancement to
  those regions.
- **UATS: Uncertainty-guided Adaptive Temperature Scaling** estimates teacher
  uncertainty and assigns sample-wise non-target temperature multipliers for
  self-knowledge distillation.

The code keeps the original self-distillation setting: ResNet18 is used as the
backbone, teacher and student branches share weights, the teacher view is
computed with `torch.no_grad()`, and only the student view is optimized.

## Method Overview

Training uses two augmented views of each image:

1. **Teacher view**: weakly augmented image, forwarded without gradient.
2. **Student view**: augmented image, optionally enhanced by ARE after warm-up.
3. **Loss**: cross entropy plus temperature-scaled KD loss.

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
    test/class0/*.jpg
    test/class1/*.jpg
  isic_k/
    train/class0/*.jpg
    train/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
  cbis/
    train/class0/*.jpg
    train/class1/*.jpg
    test/class0/*.jpg
    test/class1/*.jpg
```

You can also pass `--train_dir` and `--test_dir` to directly specify ImageFolder
paths.

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

External ImageFolder example:

```bash
python main.py --dataset isic_m --data_root ./data \
  --use_are True --use_adaptive_temp True
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

## Practical Notes

- If LARE-KD performs worse than the baseline, first try conservative settings:
  `--lambda_min 0.8 --lambda_max 1.2 --are_prob 0.5 --clahe_clip_limit 1.5`.
- Use `epoch_metrics.txt` to check whether the issue is early instability,
  overfitting, or a specific module hurting performance.
- For weak teacher predictions, try `--kd_conf_thresh 0.5` or a longer
  `--temp_warmup`.
- For noisy local enhancement, lower `--are_prob`, increase `--are_warmup`, or
  use a smaller `--clahe_clip_limit`.

## Quick Verification

```bash
python -m py_compile main.py adaptive_temp.py attention_enhance.py datapreprocess.py
```
