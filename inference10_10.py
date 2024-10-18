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
group_test.add_argument(
    "--folder_number",
    type=int,
    default=None,
    nargs="?",
    help="which folder you want to infrence",
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
FLODER_NUMBER = args.folder_number

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

# Create a new figure
plt.figure(figsize=(5, 5))


colors = [
    '#000000', '#C9655B', '#DBCF6A', '#97D869', '#88D7AD',
    '#6D98D5', '#7D58D3', '#C85EBB'
]
cmap = mcolors.ListedColormap(colors)

# Display the prediction array using matplotlib's imshow function with the 'plasma' colormap
plt.imshow(prediction, cmap=cmap, interpolation='nearest', vmin=0, vmax=7)

# Add a color bar to show the range of prediction values
cbar = plt.colorbar()
cbar.set_ticks(np.arange(0, 8, 1))
cbar.set_ticklabels([f'Class {i}' for i in range(8)])

# Add title and axis labels
plt.title(f'{img_filename}  Prediction Results')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# # Show the plot
# plt.show()

# Define the output path
output_folder = f'./VAL_patch10_10/{FLODER_NUMBER}_result/'
output_path = os.path.join(output_folder, f'{img_filename}_prediction_results.png')

# 检查并创建目录
try:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
except Exception as e:
    print(f"Error: Failed to create directory '{output_folder}'. {e}")
    raise  # 重新抛出异常，确保程序在出错时停止

# Save the plot as an image file
plt.savefig(output_path)  # 添加保存图像的命令

# Close the plot to free up memory
plt.close()

# 输出保存的文件路径
print(f'Image saved to {output_path}')