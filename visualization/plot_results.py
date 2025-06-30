"""
Plot visualization for training results from stochastic stability experiments.

This script automatically finds the most recent training run and creates
comprehensive plots showing various metrics like batch sharpness, lambda max,
batch lambda max, GNI, and accuracy.
"""

import pandas as pd
from pathlib import Path

# import plotly.express as px
from matplotlib import pyplot as plt
import numpy as np

from collections import OrderedDict

import os

# ====================================
# Configuration and Constants
# ====================================

# Column names for the results CSV file
COLUMN_NAMES = ['epoch', 'step', 'batch_loss', 'full_loss', 'batch_lmax', 'lmax', 'batch_sharpness', 'total_ghg', 'fisher_batch_eig', 'fisher_total_eig', 'batch_sharpness_static', 'gni', 'full_acc', 'param_dist']

# Check required environment variables
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

RES_FOLDER = Path(os.environ.get('RESULTS'))

# ====================================
# Data Loading and Preprocessing
# ====================================

# Find the most recently created folder
most_recent_folder = None
most_recent_time = 0

for dataset_folder in RES_FOLDER.iterdir():
    if dataset_folder.is_dir():
        for run_folder in dataset_folder.iterdir():
            if run_folder.is_dir():
                folder_mtime = run_folder.stat().st_mtime
                if folder_mtime > most_recent_time:
                    most_recent_time = folder_mtime
                    most_recent_folder = run_folder

if most_recent_folder is None:
    raise ValueError("No run folders found in the results directory")

path = most_recent_folder

print(f"Using the most recent folder: {most_recent_folder}")

# Read the specific run data
run_path = str(path)
batch_size = int(run_path.split('_')[-1][1:])  # Extract batch size from folder name
file_path = path / 'results.txt'
df = pd.read_csv(file_path, skiprows=4, sep=',', header=None, names=COLUMN_NAMES, na_values=['nan'], 
                 skipinitialspace=True)

# Extract learning rate from folder name
lr_size = run_path.split('_')[-2][2:]
lr = float(lr_size)

# Extract model name from folder path
model_name = run_path.split('/')[0].split('_')[1:]
model_name = '_'.join(model_name)

# ====================================
# Plot Setup and Configuration
# ====================================

# Create the main plot
fig, ax = plt.subplots(figsize=(10, 5))

# Add horizontal line for stability threshold (2/η)
ax.axhline(y=2/lr, color='black', linestyle='--', label=r'2/$\eta$')

# Set y-axis limits based on learning rate
ax.set_ylim(1, 2/lr*2)
# ax.set_ylim(0, 20)  # Alternative fixed limits

# ====================================
# Batch Sharpness Plotting
# ====================================

# Plot individual step sharpness values (scattered points)
ax.scatter(df['step'],
           df['batch_sharpness'], label='step sharpness', color='red', alpha=0.2, s=2)

# Calculate averaging window based on dataset size and batch size
dataset_size = 8192
steps_in_epoch = dataset_size // batch_size
average_over = steps_in_epoch * int(np.sqrt(batch_size))

# Compute rolling average of batch sharpness for smoother visualization
df['batch_sharpness_avg'] = (df['batch_sharpness']).rolling(window=average_over, min_periods=1, center=True).mean()

# Plot the averaged batch sharpness
ax.plot(df['step'], df['batch_sharpness_avg'], color='#2ca02c', label='batch sharpness', linewidth=1.5)

# ====================================
# Batch Lambda Max Plotting
# ====================================

# Plot individual step lambda max values (scattered points)
ax.scatter(df['step'],
           df['batch_lmax'], label=r'step $\lambda^b_{max}$', color='blue', alpha=0.2, s=2)

# Recalculate averaging parameters (keeping for clarity)
steps_in_epoch = dataset_size // batch_size
average_over = steps_in_epoch * int(np.sqrt(batch_size))

# Compute rolling average of batch lambda max
df['batch_lmax_avg'] = (df['batch_lmax']).rolling(window=average_over, min_periods=1, center=False).mean()

# Plot the averaged batch lambda max
ax.plot(df['step'], 
    df['batch_lmax_avg'], 
    color='black', 
    label=r'$\lambda^b_{max}$', 
    linewidth=1.5
    )

# ====================================
# Full Dataset Lambda Max Plotting
# ====================================

# Filter out NaN values for clean plotting
valid_data = df[['step', 'lmax']].dropna()

# Plot full dataset lambda max
ax.plot(valid_data['step'],
        valid_data['lmax'],
        label=r'$\lambda_{max}$',
        color='#1f77b4',
        linewidth=1.5)

# ====================================
# Gradient-Noise Interaction (GNI) Plotting
# ====================================

# # Filter out NaN values for GNI
# valid_data = df[['step', 'gni']].dropna()

# # Plot raw GNI values with transparency
# ax.plot(valid_data['step'],
#                 valid_data['gni'],
#                 label='GNI',
#                 color='#9467bd',  # Purple color
#                 alpha=0.5,
#                 linewidth=1.5)

# # Calculate smoothed GNI using rolling average for better visualization
# valid_data_smooth = df[['step', 'gni']].dropna()
# valid_data_smooth['gni_smooth'] = valid_data_smooth['gni'].rolling(window=20, min_periods=1, center=True).mean()

# # Plot smoothed GNI
# ax.plot(valid_data_smooth['step'],
#     valid_data_smooth['gni_smooth'],
#     label='GNI (smoothed)',
#     color='#9467bd',  # Same purple color
#     alpha=1,
#     linewidth=2)

# ====================================
# Plot Formatting and Labels
# ====================================

# Add vertical line for specific events (currently commented out)
# ax.axvline(x=14848, color='purple', linestyle='--', label='LR increase spot')

# Add legend and labels
ax.legend(loc='upper left') 

ax.set_title(f'batch size {batch_size}')
ax.set_xlabel('steps')

# ====================================
# Secondary Axes: Accuracy and Loss
# ====================================

# # Add third y-axis for accuracy (positioned on the right, offset outward)
# ax3 = ax.twinx()
# # Offset the position of ax3 to avoid overlap with loss axis
# ax3.spines['right'].set_position(('outward', 60))  # Move 60 points outward

# # Plot accuracy data
# valid_acc = df[['step', 'full_acc']].dropna()
# ax3.plot(valid_acc['step'], valid_acc['full_acc'], 
#          color='purple', label='accuracy',
#          marker='o', markersize=4, linestyle='-', alpha=0.7)

# # Configure accuracy axis
# ax3.set_ylabel('Accuracy')
# ax3.set_ylim(0, 1)  # Accuracy is typically between 0 and 1
# ax3.legend(loc='center right')

# Add second y-axis for loss
ax2 = ax.twinx()
ax2.plot(df['step'], df['full_loss'].interpolate(), 
         color='gray', label='full batch loss',
         alpha=1)

# Configure loss axis
ax2.set_ylabel('Loss')
ax2.set_yscale('log')  # Log scale for loss visualization
ax2.set_ylim(1e-4, 1)
ax2.legend(loc='upper right')

# ====================================
# Display Plot
# ====================================

# Save plot to img folder in the same directory as this script
script_dir = Path(__file__).parent
img_dir = script_dir / 'img'
img_dir.mkdir(exist_ok=True)

# Create filename based on the most recent folder name
plot_filename = f"{most_recent_folder.name}_results.png"
save_path = img_dir / plot_filename

plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
