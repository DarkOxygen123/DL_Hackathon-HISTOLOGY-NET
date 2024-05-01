# PreText Training and Fine-Tuning

This directory contains the code for training and fine-tuning a model using the pretext task approach. Pretext tasks are self-supervised learning tasks where both the input and the labels are derived from the input data itself. The model is first trained on a pretext task and then fine-tuned on the downstream task.

## Features

- **Pretext Task Training**: The model is initially trained on a pretext task. This involves predicting some aspect of the input data, such as the colorization of grayscale images or the reconstruction of images from their patches.

- **Fine-Tuning**: After the pretext task training, the model is fine-tuned on the downstream task. This involves training the model on the actual task of interest, here suggesting segmentation, using the weights learned from the pretext task as the initial weights.

## Files

- **PreText_training_colourize.py**: This Python script contains the code for the colorization pretext task.

- **PreText_training_PatchMasking.py**: This Python script contains the code for the Patch MAsking pretext task.

- **finetune_PreText_Model.py**: This Python script contains the code for fine-tuning the model on the downstream task. It includes the definition of the *SegmentationModel class* for representing a segmentation model using the Checkpoint weights from PreText task, and the *Trainer class* for training the model.

## Requirements
Requires the following libraries:

PyTorch
torchvision
NumPy
OpenCV
PIL

```py
class SegmentationModel:
    """
    A class representing a segmentation model.

    Args:
        model_path (str): The path to the pre-trained model.

    Attributes:
        device (torch.device): The device to be used for computation.
        model (torch.nn.Module): The segmentation model.
    """
    ...

class Trainer:
    '''
    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_dataloader : torch.utils.data.DataLoader
        The DataLoader for the training data.
    val_dataloader : torch.utils.data.DataLoader
        The DataLoader for the validation data.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    num_epochs : int
        The number of epochs to train the model.

    Methods
    -------
    train():
        Trains the model for a specified number of epochs, 
        and validates it at the end of each epoch.
    '''


