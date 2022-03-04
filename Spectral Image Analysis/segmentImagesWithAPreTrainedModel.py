# -*- coding: utf-8 -*-
"""
INDUSTRIAL PROJECT COURSE, 2021

CODE TO SEGMENT A COMPLETE DIRECTORY OF IMAGES USING A PRE-TRAINED CNN MODEL

"""

import matplotlib.pyplot as plt
from generalLibrary import  read_stiff, segmentAnImageWithTheModel
from cnnModelsLibrary import getUnetModel
import pickle
from os import walk
from dataGeneratorLibrary import createPcaMnfRgbIcaDirectories
createPcaMnfRgbIcaDirectories()

# ------------------------------ GLOBAL OPTIONS -------------------------------
    # ------------------ Selectable Parameter Through GUI ---------------------
# Choose the model:
# modelName = "UNet_With_Original_DataSet"
# modelName = "UNet_With_Naive_RGB_DataSet_Reduction"
# modelName = "UNet_With_PCA_DataSet_Reduction"
# modelName = "UNet_With_MNF_DataSet_Reduction"
# modelName = "UNet_With_ICA_DataSet_Reduction"

# Define the directory that contains the folders "datasets" and "masks":
# root_path = 'DataSets/OriginalDataSet/'
# IMAGE_CHANNELS = 38
# root_path = 'DataSets/RGBDataSet/' # Ronny, Without the naive
# IMAGE_CHANNELS = 3
# root_path = 'DataSets/PCADataSet/'
# IMAGE_CHANNELS = 3
# root_path = 'DataSets/MNFDataSet/'
# IMAGE_CHANNELS = 3
# root_path = 'DataSets/ICADataSet/'
# IMAGE_CHANNELS = 3

def segment(modelName, root_path, IMAGE_CHANNELS):
        # --------------------------- Constants -----------------------------------   
    # Choose the desired image size.
    # All the images and masks will be resized to IMAGE_SIZE x IMAGE_SIZE for better handling.
    IMAGE_SIZE = 128

    # Balance the DataSet with weights.
    #     If useWeights = True    ->      The categories will be balanced using weights
    #     If useWeights = False   ->      The categories will not be balanced.
    useWeights = True

    # ----------- Load the auxiliary variables for the desired model---------------
    # Load auxiliary variables
    # Open the file in binary mode
    with open('AuxVariables/'+'auxVar_classes_'+modelName+'.pkl', 'rb') as file:
        # Load the 'classes' variable
        classes = pickle.load(file)
    # Open the file in binary mode
    with open('AuxVariables/'+'auxVar_class_colors_'+modelName+'.pkl', 'rb') as file:
        # Load the 'classes' variable
        class_colors = pickle.load(file)

    # --------- Create the desired model and load the pretrained weights ----------
    # Create and compile the desired model
    model = getUnetModel(len(classes), IMAGE_SIZE, IMAGE_CHANNELS)
    if (useWeights == True): # With class weights
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"],sample_weight_mode="temporal") 
    else: # With no class weights
        model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"]) 

    # Load the pre-trained weights from a file
    model = getUnetModel(len(classes), IMAGE_SIZE, IMAGE_CHANNELS)
    trainedModelFileName = 'SavedModels/'+modelName+'.hdf5'
    model.load_weights(trainedModelFileName)

    # ------------- Segment and save all the image with the model -----------------
    filenames = next(walk(root_path+"Set_1_images/"), (None, None, []))[2]
    for filename in filenames:
        filenameWithoutExtension = filename[:-4]
        # print(filename)
        # Load a spectral reflectance of the image
        imageToTestName = filenameWithoutExtension
        imageToTestPath = root_path+"Set_1_images/"+imageToTestName+".tif"
        spim, wavelength, rgb_img, metadata = read_stiff(imageToTestPath)
        # Plot the RGB Original visualization
        plt.imshow(rgb_img)
        plt.title("Original_RGB_Image")
        plt.draw()
        plt.pause(0.001)

            
        # Segment the image using the trained model
        segmentAnImageWithTheModel(model, spim, IMAGE_SIZE, classes, class_colors, modelName, imageToTestName)



# segment("UNet_With_Naive_RGB_DataSet_Reduction", 'DataSets/RGBDataSet/', 3)