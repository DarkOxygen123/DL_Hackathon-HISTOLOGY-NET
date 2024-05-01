#!----IMPORTS----!#
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

np.random.seed(42)


class ImageLoader:

    """
    A class used to load labelled images and corresponding masks from given directories.
    ...

    Attributes
    ----------
    images_path : str
        a string representing the directory path of the images
    masks_path : str
        a string representing the directory path of the masks

    Methods
    -------
    load_images():
        Loads and returns all images from the images directory.
    load_masks():
        Loads and returns all masks from the masks directory.
    load_images_and_masks():
        Loads and returns all images and masks from their respective directories.
    """

    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def load_images(self):
        """
        Loads and returns all images from the images directory.

        Returns
        -------
        list of Histology images
        """
        images = []
        for filename in os.listdir(self.images_path):
            img = cv2.imread(os.path.join(self.images_path, filename))
            if img is not None:
                images.append(img)
        return images

    def load_masks(self):
        """
        Loads and returns all masks from the masks directory.

        Returns
        -------
        list of Corresponding masks
        """
        masks = []
        for filename in os.listdir(self.masks_path):
            img = cv2.imread(os.path.join(
                self.masks_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                masks.append(img)
        return masks

    def load_images_and_masks(self):
        return self.load_images(), self.load_masks()


class ImageAugmentor:
    """
    A class for performing image augmentation operations.

    Attributes:
        images (list): A list of input images.
        masks (list): A list of corresponding masks for the images.

    Methods:
        rotate_image: Rotates an image and its corresponding mask by a given angle.
        jigsaw_puzzle_image: Divides an image and its corresponding mask into patches and shuffles them.
        patch_location_prediction_image: Randomly selects a patch from an image and its corresponding mask.
        augment_image_and_mask: Augments an image and its corresponding mask by applying various operations.
    """

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    @staticmethod
    def rotate_image(image, mask, angle):
        """
        Rotates an image and its corresponding mask by a given angle.

        Args:
            image (ndarray): The input image.
            mask (ndarray): The corresponding mask for the image.
            angle (int): The angle of rotation.

        Returns:
            ndarray: The rotated image.
            ndarray: The rotated mask.
        """
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        return np.array(image), np.array(mask)

    @staticmethod
    def jigsaw_puzzle_image(image, mask):
        """
        Divides an image and its corresponding mask into patches and shuffles them.

        Args:
            image (ndarray): The input image.
            mask (ndarray): The corresponding mask for the image.

        Returns:
            ndarray: The reconstructed image.
            ndarray: The reconstructed mask.
        """
        h, w, _ = image.shape
        patches = []

        for i in range(0, h, 64):
            for j in range(0, w, 64):
                patch = image[i:i+64, j:j+64]
                mask_patch = mask[i:i+64, j:j+64]
                patches.append((patch, mask_patch))
        random.shuffle(patches)

        image_reconstructed = np.zeros((h, w, 3), dtype=np.uint8)
        mask_reconstructed = np.zeros((h, w), dtype=np.uint8)
        for i in range(0, h, 64):
            for j in range(0, w, 64):
                patch = patches.pop()
                image_reconstructed[i:i+64, j:j+64] = patch[0]
                mask_reconstructed[i:i+64, j:j+64] = patch[1]

        return image_reconstructed, mask_reconstructed

    @staticmethod
    def patch_location_prediction_image(image, mask):
        """
        Randomly selects a patch from an image and its corresponding mask.

        Args:
            image (ndarray): The input image.
            mask (ndarray): The corresponding mask for the image.

        Returns:
            ndarray: The selected patch from the image.
            ndarray: The selected patch from the mask.
        """
        h, w, _ = image.shape
        x = random.randint(0, w-128)
        y = random.randint(0, h-128)
        image = image[y:y+128, x:x+128]
        mask = mask[y:y+128, x:x+128]
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        return image, mask

    def augment_image_and_mask(self, image, mask):
        """
        Augments an image and its corresponding mask by applying various operations.

        Args:
            image (ndarray): The input image.
            mask (ndarray): The corresponding mask for the image.

        Returns:
            list: A list of augmented images.
            list: A list of augmented masks.
        """
        augmented_images = [np.array(image)]
        augmented_masks = [np.array(mask)]
        for i in range(7):
            augmented_image = image
            augmented_mask = mask
            if random.random() < 0.4:
                angle = random.randint(0, 360)
                augmented_image, augmented_mask = self.rotate_image(
                    augmented_image, augmented_mask, angle)
            if random.random() < 0.2:
                augmented_image, augmented_mask = self.jigsaw_puzzle_image(
                    augmented_image, augmented_mask)
            if random.random() < 0.2:
                augmented_image, augmented_mask = self.patch_location_prediction_image(
                    augmented_image, augmented_mask)
            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)

        return augmented_images, augmented_masks


class ImageSaver:
    def __init__(self, dir_name, dir_name1, dir_name2):
        self.dir_name = dir_name
        self.dir_name1 = dir_name1
        self.dir_name2 = dir_name2
        self.create_directories()

    def create_directories(self):
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        if not os.path.exists(self.dir_name1):
            os.makedirs(self.dir_name1)
        if not os.path.exists(self.dir_name2):
            os.makedirs(self.dir_name2)

    def save_images(self, images, augmented_images, augmented_masks):
        for i in tqdm(range(len(images)), desc="Processing images"):
            cv2.imwrite(f'{self.dir_name2}/' + str(i) + '.jpeg', images[i])
            for j in range(len(augmented_images)):
                cv2.imwrite(f'{self.dir_name}/' + str(i) + '_' +
                            str(j)+'.jpeg', augmented_images[j])
            for j in range(len(augmented_masks)):
                cv2.imwrite(f'{self.dir_name1}/' + str(i) +
                            '_' + str(j)+'.jpeg', augmented_masks[j])


if __name__ == "__main__":
    images_path = 'images1'  # Path to the images directory
    masks_path = 'labels1'   # Path to the masks directory

    loader = ImageLoader(images_path, masks_path)
    images, masks = loader.load_images_and_masks()

    augmentor = ImageAugmentor(images, masks)

    # Directory names to save the augmented images, masks and original images
    # If not already present will be created
    dir_name = "final_images"  # Name of Directory to save the augmented images
    dir_name1 = "final_masks"  # Name of Directory to save the augmented masks
    # Name of Directory to save the original images
    dir_name2 = "final_original_images"
    saver = ImageSaver(dir_name, dir_name1, dir_name2)

    for i in range(len(images)):
        augmented_images, augmented_masks = augmentor.augment_image_and_mask(
            images[i], masks[i])
        saver.save_images(images, augmented_images, augmented_masks)
