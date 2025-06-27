# Edge of Stochastic Stability (EOSS)

This repository contains research code for exploring the **Edge of Stochastic Stability** during neural network training. It includes a configurable training script, measurement utilities and analysis helpers for evaluating sharpness, gradient noise interaction and related metrics.

## Installation

1. Create a virtual environment (optional):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset & Results Folders

The scripts use environment variables to locate dataset and results directories. Set the following environment variables before running any scripts:

```bash
export DATASETS="/path/to/your/datasets"
export RESULTS="/path/to/your/results"
```

If these environment variables are not set, the scripts will fall back to default paths:
- **Default datasets path**: `/scratch/gpfs/andreyev/datasets/`
- **Default results path**: `/scratch/gpfs/andreyev/eoss/results`

The following scripts use these environment variables:
- `training.py`
- `compute_final_fullbs.py` 
- `compute_final_gHg.py`
- `sharpness_gap.py`

Example structure:
```
$DATASETS/
├── cifar10/
└── other_datasets/

$RESULTS/
├── experiment1/
└── experiment2/
```

## Running Training

A typical invocation looks like:
```bash
export DATASETS="/path/to/datasets"
export RESULTS="/path/to/results"
python training.py --dataset cifar10 --model mlp --batch 64 --epochs 20 --lr 0.001
```
The script logs metrics to `results.txt` inside a timestamped folder under the results directory. Checkpoints are stored in `checkpoints/`.

### Measurement Options

Training supports additional measurements such as Lambda Max, batch sharpness and Gradient-Noise Interaction. Enable them with flags like `--lambdamax`, `--batch-sharpness` and `--gni`. See `training.py --help` for all options.

## Post-Training Analysis

After training finishes, you can compute final metrics with:
```bash
export DATASETS="/path/to/datasets"
export RESULTS="/path/to/results"
#TODO
```
These scripts expect the same dataset and results directories as the training script (configured via environment variables).


## Visualization

A lightweight `plot_results.py` script (to be added) will generate simple plots from `results.txt` using Matplotlib.

## Directory Layout

```
training.py              # main training entry point
utils/                   # data loading, models and measurement utilities
requirements.txt         # dependencies
README.md                # this file
```

## License

Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



