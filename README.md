# Edge of Stochastic Stability (EOSS)

This repository contains the research code accompanying the paper [Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD](https://arxiv.org/abs/2412.20553) by Arseniy Andreyev and Pierfrancesco Beneventano

## Installation

1. Create a virtual environment (optional; you can use conda instead):
   ```bash
   python3 -m venv eoss
   source eoss/bin/activate
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

The following scripts use these environment variables:
- `training.py`
- `download_datasets.py`
- `visualization/plot_results.py`


### Downloading the datasets

Before the first training run, download the datasets:
```bash
python download_datasets.py
```


Example structure:
```
$DATASETS/
├── cifar10/
└── other_datasets/

$RESULTS/
├── cifar10_mlp/
│   └── 20250625_0640_14_lr0.00800_b8/
│       ├── results.txt
│       └── checkpoints/
└── other_experiments/
```

## Running Training

A typical invocation looks like:
```bash
python training.py --dataset cifar10 --model mlp --batch 8 --lr 0.01 --steps 150000  --num_data 8192 --init_scale 0.2 --dataset_seed 111 --init_seed 8312 --lambdamax --batch-sharpness
```

This command:
- Trains an MLP on CIFAR-10 with 8192 samples
- Uses batch size 8 and learning rate 0.01
- Runs for 150,000 steps
- Initializes weights with scale 0.2
- Tracks Lambda Max (full-batch sharpness) and batch sharpness during training

The last set of flags (`--lambdamax`, `--batch-sharpness`, etc.) specify which quantities to track during training. Without these flags, only basic metrics like loss and accuracy are recorded.

The script logs metrics to `results.txt` inside a timestamped folder under the results directory. Checkpoints are stored in `checkpoints/`.

### Measurement Options

Training supports additional measurements such as lambda max, batch sharpness, batch lambda max and Gradient-Noise Interaction. See `training.py --help` for all options.

## Post-Training Analysis

After training finishes, you can visualize the metrics with:
```bash
python visualization/plot_results.py
```
The `visualization/plot_results.py` script automatically finds and plots the most recent training run in the RESULTS folder, generating simple plots from `results.txt` and saving them in the `visualization/img/` folder.
See `visualization/data_visualization.ipynb` for an example of a notebook to plot additional quantities and customize the plots

See also `experiments/template.ipynb` for a template jupyter notebook to run further experiments on trained networks

## Directory Layout

```
edge-of-stochastic-stability/
├── training.py                      # main training entry point
├── download_datasets.py             # script to download required datasets
├── requirements.txt                 # Python dependencies
├── README.md                        # this file
├── LICENSE                          # Apache 2.0 license
├── .gitignore                       # Git ignore rules
├── utils/                           # core utilities
│   ├── __init__.py
│   ├── data.py                      # dataset loading and preprocessing
│   ├── nets.py                      # neural network models (MLP, CNN, ResNet)
│   ├── measure.py                   # procedures to measure sharpness and other quantities
│   ├── noise.py                     # utilities for running noisy GD
│   ├── storage.py                   # result storage and checkpointing
│   ├── resnet_new.py               # ResNet model implementations
│   ├── resnet_bn.py                # ResNet with batch normalization
│   └── resnet_bn_new.py            # Updated ResNet with batch norm
├── visualization/                   # plotting and analysis tools
│   ├── plot_results.py             # example script to generate plots from results
│   ├── data_visualization.ipynb    # template for additional plots of runs
│   └── img/                        # generated plots and figures
└── experiments/                     # experiment templates and notebooks
    └── template.ipynb              # template jupyter notebook for experiments
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



