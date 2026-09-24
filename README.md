# Swap-Adversarial Framework (SAF)

PyTorch implementation for **A Swap-Adversarial Framework for Improving Domain Generalization in Electrocorticography-Based Parkinson's Disease Classification**.

We propose a Swap-Adversarial Framework (SAF) for domain generalization in ECoG-based Parkinson's disease classification. SAF combines Inter-Subject Balanced Channel Swap (ISBCS) with domain-adversarial learning (DAL) to encourage class-relevant features while reducing subject-specific information.

## Installation

Tested with Python 3.13.11 and PyTorch 2.10.0. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Data

Prepare preprocessed signals as `.npy` arrays of shape **[samples, channels, time points]**, with matching channel and time dimensions across files.

Copy [config/data.example.json](config/data.example.json) to `config/data.json` and replace its example paths and labels with your own:

- `data_list` assigns files to class labels within each split.
- `domain_list` assigns the same files to subject IDs within each split.
- Paths are relative to `--data-root`. A file must belong to only one split.

## Training

```bash
python train.py \
  --data-config config/data.json \
  --data-root /path/to/windows \
  --output-dir runs/saf
```

## Code

| Directory | Contents |
| --- | --- |
| `src/data/` | Dataset loading, splits, and ISBCS |
| `src/models/` | EEGNet and DAL losses |
| `src/training/` | Training, metrics, and checkpoints |
| `src/utils/` | Gradient reversal |
