# HISTOLOGY-NET
This project is a result of the Hackathon conducted by Yamaha and IIT Mandi. It contains various models and utilities developed for the purpose of the Hackathon.

## Project Description
The problem statement for the Hackathon was to develop a self-supervised learning model for efficient binary segmentation of histology images, with little labelled data and abundant unlabelled data; and integrate it into a user-friendly webpage for practical utility. This project aims to address this problem and provide a solution that meets the requirements.

## Folder Structure
* `Main_models`: This directory contains the primary models used in the project. Each model is housed in its own subdirectory, with the model's name indicating its architecture or the specific task it was designed for.

* `Image_augmentation`: This directory contains code for Image augmentation which was used to train models and allowing us to increase the overall trainable data

* `Imports`: This directory contains functions that are used throughout the project

* `PreText_Training_FineTuning`: This directory contains code for PreText Task based training using Augmented images and subsequent finetuning using labelled image-mask pairs as provided (which were also augmented for increaseing training data)

Further detains of imports and code informations are given within each of the directories, furthermore all the code files are commented and incase of any missing files or discrepencies feel free to contact us.

## Performance
Our current best model is the Ensemble model attached to the sequential model, which achieved a 
*   Dice score: **0.8388**
*   Jaccard score: **0.7239**

## Contributors
This project was made possible by the hard work and dedication of

* Dhruv
* Uthamkumar M
* Shashank Dwivedi
* Paras Bedi
* Riya Arora

## Acknowledgements

We would like to express our deepest gratitude to Yamaha and IIT Mandi for conducting this Hackathon in the duration of the course Deep Learning and its Applications. It was a great opportunity for us to learn and grow as developers.