# -*- coding: utf-8 -*-
"""
INDUSTRIAL PROJECT COURSE, 2021

MAIN CODE TO TRAIN AND TEST THE UNET CNN SEGMENTATION MODEL

"""

from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, jaccard_score, ConfusionMatrixDisplay
from generalLibrary import loadImagesAndMasks, saveFileWithClassAndColorIdentifiers, reEncodeTheMasksFormat, calculateSampleWeightsToBalanceTheDataSet, plotIoUBarGraphPerClass
from cnnModelsLibrary import getUnetModel
from dataGeneratorLibrary import createPcaMnfRgbIcaDirectories
from keras.metrics import MeanIoU
import random
import pickle
import time

createPcaMnfRgbIcaDirectories()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# ------------------------------ GLOBAL OPTIONS -------------------------------
    # ------------------ Selectable Parameter Through GUI ---------------------
# Choose the model:
# modelName = "UNet_With_Original_DataSet"
# modelName = "UNet_With_Naive_RGB_DataSet_Reduction"
# modelName = "UNet_With_PCA_DataSet_Reduction"
# modelName = "UNet_With_MNF_DataSet_Reduction"
modelName = "UNet_With_ICA_DataSet_Reduction"

# Define the directory that contains the folders "datasets" and "masks":
# root_path = 'DataSets/OriginalDataSet/'
# IMAGE_CHANNELS = 38
# root_path = 'DataSets/RGBDataSet/' # Ronny, Without the naive
# IMAGE_CHANNELS = 3
# root_path = 'DataSets/PCADataSet/'
# IMAGE_CHANNELS = 3
# root_path = 'DataSets/MNFDataSet/'
# IMAGE_CHANNELS = 3
root_path = 'DataSets/ICADataSet/'
IMAGE_CHANNELS = 3

# Training Mode:
#     If trainTheModelFlag = True    ->      The model will be trained from scratch
#     If trainTheModelFlag = False   ->      The model will load previous weights
# trainTheModelFlag = True
trainTheModelFlag = False

    # --------------------------- Constants -----------------------------------
# Choose the desired image size.
# All the images and masks will be resized to IMAGE_SIZE x IMAGE_SIZE for better handling.
IMAGE_SIZE = 128

# Balance the DataSet with weights.
#     If useWeights = True    ->      The categories will be balanced using weights
#     If useWeights = False   ->      The categories will not be balanced.
useWeights = True

# Name of the file with the previously saved model (if it exists)
trainedModelFileName = 'SavedModels/'+modelName+'.hdf5'

# Name of the file to save the classes with their correspoding colors (Optional)
classAndColorFileName = "classAndColor.npy"
# 12 Color Codes Available For The Segmentation
class_colors = np.array([[0,0,0],         # Background
                          [0,255,255],     # Blue dye
                          [0,255,0],       # ICG
                          [191,91,22],     # Specular reflection
                          [237,28,36],     # Artery
                          [63,72,204],     # Vein
                          [128,128,128],   # Stroma
                          [34,177,76],     # Artery, ICG
                          [255,255,255],   # Stroma, ICG
                          [128,128,0],     # Suture
                          [255,242,0],     # Umbilical cord
                          [128,0,0]])      # Red dye

# --------------------------------- DATA LOADING ------------------------------ 
X,y, classes = loadImagesAndMasks(root_path, IMAGE_SIZE)

# Save the classes with their correspoding colors in a file for later use
saveFileWithClassAndColorIdentifiers(classAndColorFileName, classes, class_colors)

# Optional Plot of Classes Distributions
plt.figure(1)
np.unique(np.array(y))
plt.hist(y.flatten(), bins=np.arange(-1,12)+0.5, ec="k")
plt.xticks(np.arange(12), classes, rotation='vertical')  # Set text labels.
plt.title("Original_Classes_Distributions")
plt.savefig('TrainingPlots/'+ modelName +'/'+'Original_Classes_Distributions.png', bbox_inches='tight')
plt.show()

# Divide the data set into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
y_train_cat = to_categorical(y_train)

# OPTIONAL
# Due to the unbalanced data set, calculate weights for the classes.
if (useWeights == True): # With class weights
    y_reshaped = reEncodeTheMasksFormat(y, classes)
    sample_weights = calculateSampleWeightsToBalanceTheDataSet(y_reshaped, y_train)


# ----------------------- DEEP LEARNING MODEL CREATION ------------------------
model = getUnetModel(len(classes), IMAGE_SIZE, IMAGE_CHANNELS)
if (useWeights == True): # With class weights
    model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"],sample_weight_mode="temporal") 
else: # With no class weights
    model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"]) 
model.summary()

# ------------------------IF: WE WANT TO TRAIN THE MODEL ----------------------
if trainTheModelFlag == True:
    # Model training
    start = time.time() # Start measuring the training time
    if (useWeights == True): # With class weights
        history = model.fit(X_train,y_train_cat,batch_size=2,epochs=350,sample_weight=sample_weights)
    else: # With no class weights
        history = model.fit(X_train,y_train_cat,batch_size=2,epochs=350) 
    stop = time.time() # Start measuring the training time
    print(f"Total Training time: {stop - start} Seconds")

    # Save the model in a file
    model.save(trainedModelFileName)
    # Save Other Auxiliary Variables
        # Open a file and use dump() to save the 'classes' variable
    with open('AuxVariables/'+'auxVar_classes_'+modelName+'.pkl', 'wb') as file:
        pickle.dump(classes, file)
        # Open a file and use dump() to save the 'classes' variable
    with open('AuxVariables/'+'auxVar_class_colors_'+modelName+'.pkl', 'wb') as file:
        pickle.dump(class_colors, file)
    
    # OPTIONAL
    # Plot the training loss at each epoch of the training
    plt.figure(2)
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.title(modelName + ' Train_Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('TrainingPlots/'+ modelName +'/'+'Train_Loss.png', bbox_inches='tight')
    plt.show()
    
    # Plot the training accuracy at each epoch of the training
    plt.figure(3)
    acc = history.history['accuracy']
    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.title(modelName + ' Train_Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('TrainingPlots/'+ modelName +'/'+'Train_Accuracy.png', bbox_inches='tight')
    plt.show()
    
    # ----------------------- Final Accuracy (Train) --------------------------
    print(modelName + " Final Accuracy (Train): " + str(acc[-1]))
    

# ------------------- ELSE: WE WANT TO LOAD A PRETRAINED MODEL ----------------
else:
    # Load the model from a file
    model.load_weights(trainedModelFileName)

# ------------------- Now, let's predict in the Test Set ----------------------
y_preds = model.predict(X_test)
y_preds = np.argmax(y_preds,axis=3)
flattened_y_preds = y_preds.flatten()
flattended_y_test = y_test.flatten()

# ----------------------- Mean Accuracy (Test) --------------------------------
meanAccuracy = accuracy_score(flattened_y_preds,flattended_y_test)
print(modelName + " Mean Accuracy (Test): " + str(meanAccuracy))

# --------------------------- Confusion Matrix --------------------------------
plt.figure(4)
#disp = ConfusionMatrixDisplay.from_predictions(flattended_y_test, flattened_y_preds, display_labels=classes[:-1], include_values=False, xticks_rotation= 'vertical')
disp = ConfusionMatrixDisplay.from_predictions(flattended_y_test, flattened_y_preds, include_values=False, normalize='true')
plt.title(modelName + " Confusion_Matrix (Test): ")
plt.savefig('TrainingPlots/'+ modelName +'/'+'Confusion_Matrix.png', bbox_inches='tight')
plt.show()

# --------------- Intersection Over Union (IoU) Per Class ---------------------
IoU = jaccard_score(flattened_y_preds,flattended_y_test,average=None)
print(modelName + " IoU (Test): ")
print(IoU)
plt.figure(5)
plotIoUBarGraphPerClass(IoU, classes, modelName)
plt.show()

# --------------- Mean Intersection Over Union (IoU) --------------------------
numberOfEffectiveClassesForIoU = IoU.shape[0] 
# This value "numberOfEffectiveClassesForIoU" is always less or equal than the total number of possible "classes" = 12. 
# (This is an issue due to the very small data set available for training)
# Usually, numberOfEffectiveClassesForIoU will be 9. 
meanIoUValue = MeanIoU(num_classes = numberOfEffectiveClassesForIoU)
meanIoUValue.update_state(flattended_y_test, flattened_y_preds)
print(modelName + " Mean IoU (Test): " + str(meanIoUValue.result().numpy()))



