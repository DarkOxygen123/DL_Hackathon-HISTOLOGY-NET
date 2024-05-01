# Image Augmentation Tool

This tool is designed to perform various image augmentation operations on a set of images and their corresponding masks. The operations include rotation, jigsaw puzzle effect, and patch location prediction.

## Classes

The tool consists of three main classes:

1. `ImageLoader`: This class is used to load images and their corresponding masks from given directories.

2. `ImageAugmentor`: This class performs the image augmentation operations. It can rotate an image, apply a jigsaw puzzle effect, and select a random patch from an image.

3. `ImageSaver`: This class is used to save the original and augmented images and masks to specified directories. If the directories do not exist, they are created.

## Usage

To use this tool, you need to specify the paths to the directories containing the images and masks. You also need to specify the names of the directories where the original and augmented images and masks will be saved. All this is to be given in the __name__ == "__main__": part at the end

Here is an example:

```python
images_path = 'images1'  # Path to the images directory
masks_path = 'labels1'   # Path to the masks directory

loader = ImageLoader(images_path, masks_path)
images, masks = loader.load_images_and_masks()

augmentor = ImageAugmentor(images, masks)

dir_name = "final_images"  # Directory to save the augmented images
dir_name1 = "final_masks"  # Directory to save the augmented masks
dir_name2 = "final_original_images"  # Directory to save the original images
saver = ImageSaver(dir_name, dir_name1, dir_name2)

for i in range(len(images)):
    augmented_images, augmented_masks = augmentor.augment_image_and_mask(images[i], masks[i])
    saver.save_images(images, augmented_images, augmented_masks)

