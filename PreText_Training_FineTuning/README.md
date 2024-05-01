# PreText Training and Fine-Tuning

This folder involves training and fine-tuning a model using the pretext task approach. Pretext tasks are self-supervised learning tasks where both the input and the labels are derived from the input data itself (Here denotes the unlabelled images). The model is first trained on a pretext task and then fine-tuned on the downstream task (here denoting the Segmentation Task).


## Features

- **Pretext Task Training**: The model is initially trained on a pretext task. This involves predicting some aspect of the input data, such as the colorization of grayscale images or the reconstruction of images from their patches.

- **Fine-Tuning**: After the pretext task training, the model is fine-tuned on the downstream task. This involves training the model on the actual task of interest, here suggesting segmentation, using the weights learned from the pretext task as the initial weights. The encoder is kept the same with a different decoder while the new UNet is trained on labelled Images and correspondign masks. (With the pretrained Encoder helping extract goos features and aiding in faster training times.)

## Files
'pretext-all_models_concatenated.ipynb' This Jupyter notebook contains the code for training the model on the pretext task and concatenating the outputs of all models.

'pretext-colourize (2).ipynb' This Jupyter notebook contains the code for the colorization pretext task.

'training_second_stage_model_sequential.ipynb' This Jupyter notebook contains the code for fine-tuning the model on the downstream task. You can add any pretrained checkpoint to this file.

In case any new modified PreTraining Tasks need to be run, then modifying any of the PreText training files can be easily done by changing the first preprocessing function alone.

## Requirements
Requires the following libraries:

PyTorch
torchvision
NumPy
OpenCV
PIL
