# Main_models
The `Main_models` directory contains the primary models used in this project. Each model is housed in its own subdirectory, with the model's name indicating its architecture or the specific task it was designed for.

## Structure
Subdirectory within Main_models contains the following files:

* `1_basic_UNET_Supervised`: First model trained with 10x multi augmented labelled data without preText Training.

* `2_Sequential_model`: Made for finer detail removals from model outputs. was tested with outputs from **1_basic_UNET_Supervised** and is used as the final step in **3_Ensemble_model**

* `3_Ensemble_model`: Final Model using 3 PreText trained and finetuned models whose outputs are joinedtogether to get final mask output. For slightly better results the reuslting outputs is passed through a sequential model to remodel any small noises in patch that crept in.