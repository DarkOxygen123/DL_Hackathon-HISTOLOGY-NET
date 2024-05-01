# Imports Folder

This folder contains various utility modules that are used across the project.

## Modules

1. `loss.py`: This module contains the implementation of the `DiceLoss` class, which is a custom loss function used in the training of the neural networks in this project. The `DiceLoss` class extends PyTorch's `nn.Module` and overrides the `forward` method to calculate the Dice Loss between the predicted input and the target.

2. `preTrain_UNET.py`: This module contains the implementation of the `UNet11` and `UNet16` classes, which are the U-Net models used in this project. The U-Net is a type of convolutional neural network that is widely used for biomedical image segmentation. Depending on your customizations use Either of them can be used in place of the other.

## Usage

To use the modules in this folder, simply import them in your Python scripts. For example:

```python
from Imports.loss import DiceLoss
from Imports.preTrain_UNET import UNet11, UNet16