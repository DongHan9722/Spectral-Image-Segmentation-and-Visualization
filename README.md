# Spectral Image Analysis 

This file is to clarify the usage of our codes & files in Industrial Project Course.

## Description

In the code folder, there are segmentation part and GUI part

## Getting Started

### Dependencies

* Have Python installed.
* OS version: Mac, Windows 10, Linux

### Installing

* Download the folder in OneDrive 
* The input spectral images (in tif format) should put into path:'\DataSets\OriginalDataSet\Set_1_images'
* The input mask (in tif format) should put into path:'\DataSets\OriginalDataSet\Set_1_masks'
* Put DataSets folder inside of the folder which is downloaded from OneDrive

### Environment setting

* Using ```pipenv```

Creating python dev environment and installing dependencies:

```
pipenv install
```

Running codes:
```
pipenv run
```

---
If using the ```conda```
* Creating a New Environment 

```
conda create -n IDP_COSI python=3.8.8
```
* Activate  Environment

For  Windows :
```
conda activate IDP_COSI 
```
For  Mac :
```
conda activate IDP_COSI 
```

---

* Installing a Python Kernel
```
pip install ipykernel
```

---

* Using pip

```
pip install -r requirements.txt
```

### Executing program

* Note: **First time to Run the Code.**
When runing the main_gui.py, the new datasets (RGB, PCA, MNF, ICA) will be generated automatically in the same path as original data. This process will take from couple minutes to more than 10 minutes (depending on the hardware). When it finishes, the final size of folder **DataSets** will go up to 7.27GB.
Those datasets will be used for GUI method functions: U-Net-RGB, U-Net-PCA, U-Net-MNF, U-Net-ICA.

* Run the GUI Software 
```
python main_gui.py
```
* Run Without Generate the New Datasets
If you want run GUI without generating the new datasets mentioned above, and don’t care to use U-Net-RGB, U-Net-PCA, U-Net-MNF, U-Net-ICA:

* Comment the line below in file **segmentImagesWithAPreTrainedModel.py**
```
createPcaMnfRgbIcaDirectories()
```
* Then Run 
```
python main_gui.py
```

## File Explanation 


### Segmentation part

- generalLibrary .py
    - This is the general library that contains different functions and methods for this project.
- cnnModelsLibrary .py
    - This contains the definition of the architecture of the original UNet Model. Source: arXiv:1505.04597.
- dataGeneratorLibrary .py
    - This is the library that contains the function to generate the PCA, RGB, MNF, and ICA versions of the original data set.
    - This code is executed automatically when running "mainTrainAndTest" or "segmentImagesWithAPreTrainedModel "
- mainTrainAndTest .py
    - Code used to train and test the UNet CNN segmentation models (All the variants: Hypercube, PCA, MNF, ICA, RGB).
    - The input images should be placed in this folder: '\DataSets\OriginalDataSet\Set_1_images', and they should be in .tif format.
    - The input masks should be placed in this folder: '\DataSets\OriginalDataSet\Set_1_masks', and they should be in .tif format with the same name as their corresponding images.
- segmentImagesWithAPreTrainedModel .py
    -  Code is used to segment a complete directory of images using a pre-trained CNN model.
    -  The input images should be placed in this folder '\DataSets\OriginalDataSet\Set_1_images' and they should be in .tif format.
    -  Then, the segmentation output will be saved in this folder '\SegmentationResults'.
- SavedModels
    - This folder is automatically generated after executing "mainTrainAndTest".
    - It contains the weights of the segmentation networks.
    - Then, those weights will be loaded while running "segmentImagesWithAPreTrainedModel ".
- utils
    - This folder contains the auxiliary function "spectral_tiffs" and "vec_spim" used in "dataGeneratorLibrary"
- AuxVariables
    -  This folder contains other auxiliary variables generated while running "mainTrainAndTest".
- TrainingPlots
    -  This folder contains some graph generated during the training 
    
### Segmentation part 2

- spectral_tiffs.py 
    - Functions to read and write spectral image data
- dataPrep_random.py
    - Contains functions to read spectral image and mask data, downscale the images into smaller size acceptable by the DL networks, and save numerical data in generated .npy files in ```/data_arrays``` directory
- generate_image.py
    - A python script designed to read input image data, and output segmentations using pre-trained netwrok weights saved in ```/saved_model``` directory
    - For a standalone use, run:
        - ```python generate_image.py --model 'INSERT_MODEL_NAME' --img_dr 'PATH_TO_IMAGES'```
        - Possible choice between 'unet3plus' and 'swinUnet' architectures
        - Results saved into ```/output_images```
- for_gui.py
    - An adaptation of the ```generate_image.py``` script for GUI implementation
- train. py
    - Code to train 'unet3plus' and 'swinUnet' models. Outputs plotted training progress into ```/training_results``` and saves model weights into ```/saved_models```
    - For a standalone use, run:
        - ```python train.py --model 'INSERT_MODEL_NAME' --img_dr 'PATH_TO_IMAGES' --mask_dir 'PATH_TO_MASKS'```
- eval. py
    - Code to evaluate a trained model. Outputs confusion matrices, metrics scores, and IoU barplots into ```/training_results```
    - For a standalone use, run:
        - ```python train.py --model 'INSERT_MODEL_NAME' --img_dr 'PATH_TO_IMAGES' --mask_dir 'PATH_TO_MASKS```
- unet3plus_complete_workflow.sh
    - Shell script to train a 'unet3plus' model, evaluate and generate segmented images
- swinUnet_complete_workflow.sh
    - Shell script to train a 'swinUnet' model, evaluate and generate segmented images
    
        
 

### Requirement file

- requirements.txt
   -This requirements file includes all the  **python packages are required to run the project**graphs and plots generated after training the UNet segmentation models.

### Data

- DataSets
   -The input data needed for segmentation in this folder.

### Results

- SegmentationResults
   -The segmentation results saved in this folder

### GUI part

- UI
   -In this folder, there are several file about layout of GUI

- main_gui.py
   -This is main python file to run the GUI

### User Guide
<img width="763" alt="user_guide" src="https://user-images.githubusercontent.com/80296065/141322987-cced013e-7dca-480c-b41c-100c8ec0b0ad.png">


## Authors

Dong Han

Ronny Velastegui Sandoval

Linna Yang

Zolbayar Shagdar

Maria Jose Rueda Montes


