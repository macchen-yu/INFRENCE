# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division
import joblib
import os
from utils import convert_to_color_, convert_from_color_, get_device
from datasets import open_file
from models import get_model, test
import numpy as np
import seaborn as sns
from skimage import io
import argparse
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    build_dataset,
    show_results,
    compute_imf_weights,
    get_device,
)
# Test options
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Model to train. Available:\n"
    "SVM (linear), "
    "SVM_grid (grid search on linear, poly and RBF kernels), "
    "baseline (fully connected NN), "
    "hu (1D CNN), "
    "hamida (3D CNN + 1D classifier), "
    "lee (3D FCN), "
    "chen (3D CNN), "
    "li (3D CNN), "
    "he (3D CNN), "
    "luo (3D CNN), "
    "sharma (2D CNN), "
    "boulch (1D semi-supervised CNN), "
    "liu (3D semi-supervised CNN), "
    "mou (1D RNN)",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=-1,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)

group_test = parser.add_argument_group("Test")
group_test.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)
group_test.add_argument(
    "--image",
    type=str,
    default=None,
    nargs="?",
    help="Path to an image on which to run inference.",
)
group_test.add_argument(
    "--only_test",
    type=str,
    default=None,
    nargs="?",
    help="Choose the data on which to test the trained algorithm ",
)
group_test.add_argument(
    "--mat",
    type=str,
    default=None,
    nargs="?",
    help="In case of a .mat file, define the variable to call inside the file",
)
group_test.add_argument(
    "--n_classes",
    type=int,
    default=None,
    nargs="?",
    help="When using a trained algorithm, specified  the number of classes of this algorithm",
)
# Training options
group_train = parser.add_argument_group("Model")
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
MODEL = args.model
# Testing file
MAT = args.mat
N_CLASSES = args.n_classes
INFERENCE = args.image
TEST_STRIDE = args.test_stride
CHECKPOINT = args.checkpoint

img_filename = os.path.basename(INFERENCE)
basename = MODEL + img_filename
dirname = os.path.dirname(INFERENCE)

img = open_file(INFERENCE)
if MAT is not None:
    img = img[MAT]
# last_key = list(img.keys())[-1]
# img = img[last_key]
# Normalization
img = np.asarray(img, dtype="float32")
# img = (img - np.min(img)) / (np.max(img) - np.min(img))
N_BANDS = img.shape[-1]
hyperparams = vars(args)
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "device": CUDA_DEVICE,
        "ignored_labels": [0],
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", N_CLASSES)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


if MODEL in ["SVM", "SVM_grid", "SGD", "nearest"]:
    model = joblib.load(CHECKPOINT)
    w, h = img.shape[:2]
    X = img.reshape((w * h, N_BANDS))
    prediction = model.predict(X)
    prediction = prediction.reshape(img.shape[:2])
else:
    model, _, _, hyperparams = get_model(MODEL, **hyperparams)
    model.load_state_dict(torch.load(CHECKPOINT))
    probabilities = test(model, img, hyperparams)
    prediction = np.argmax(probabilities, axis=-1)

prediction = prediction[3:-3, 3:-3]

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import matplotlib.colors as mcolors

# Load the ground truth image
gt = np.load(os.path.join(dirname, "groundtruth", f"gt_{img_filename}"))

# Calculate metrics and print them manually
run_results = metrics(prediction, gt, n_classes=N_CLASSES)

# Create a new figure for metrics visualization, confusion matrix, and prediction result
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

# Plotting the prediction result
colors = [
    '#000000', '#C9655B', '#DBCF6A', '#97D869', '#88D7AD',
    '#6D98D5', '#7D58D3', '#C85EBB'
]
cmap = mcolors.ListedColormap(colors)

im2 = ax3.imshow(prediction, cmap=cmap, interpolation='nearest', vmin=0, vmax=7)

# Assuming class names
class_names = ['background', 'T100', 'C100', 'C20T80', 'C80T20', 'C65T35', 'C35T65', 'C50T50']

# Draw the color bar for prediction
cbar = plt.colorbar(im2, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_ticks(np.arange(0, 8, 1))  # Setting ticks
cbar.set_ticklabels(class_names)  # Using class names to replace numbers

# Add title and axis labels
ax3.set_title(f'{INFERENCE} Prediction Results')
ax3.set_xlabel('X-axis')
ax3.set_ylabel('Y-axis')

# Plotting Confusion Matrix with class names
if 'Confusion matrix' in run_results:
    confusion_matrix = run_results['Confusion matrix']
    im1 = ax1.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names)

    # Add text annotations in each cell of the confusion matrix
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            ax1.text(j, i, f'{confusion_matrix[i][j]}', ha='center', va='center', color='black')

    # Draw the color bar for the confusion matrix
    plt.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)

# Manually output each item from the metrics dictionary
text_metrics = ""
for key, value in run_results.items():
    if key == 'Confusion matrix':
        continue
    elif isinstance(value, list):
        formatted_values = [f"{item:.4f}" if isinstance(item, (int, float)) else str(item) for item in value]
        text_metrics += f"{key}: [{', '.join(formatted_values)}]\n"
    else:
        text_metrics += f"{key}: {value:.4f}\n" if isinstance(value, (int, float)) else f"{key}: {value}\n"

# Displaying the metrics in a text box in the same figure
props = dict(boxstyle='round,pad=1.5', facecolor='wheat', alpha=0.5)
ax2.text(0.5, 0.5, text_metrics, transform=ax2.transAxes, fontsize=12,
          verticalalignment='center', horizontalalignment='center', bbox=props)
ax2.axis('off')
ax2.set_title('Run Metrics')

# Show the combined plot
plt.tight_layout()
plt.show()
