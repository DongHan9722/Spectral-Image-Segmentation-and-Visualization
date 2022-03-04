# -*- coding: utf-8 -*-
"""
INDUSTRIAL PROJECT COURSE, 2021

Library that contains different functions and methods for this project.

Note: The functions read_stiff and read_mtiff were taken from the previous code of IDP.

"""

#Import Libraries
import warnings
import numpy as np
from tifffile import TiffFile
import matplotlib.pyplot as plt
from skimage.transform import resize
from os import listdir
from sklearn.utils import class_weight
from matplotlib.patches import Rectangle
import os
from PIL import Image



def read_stiff(filename: str, silent=False, rgb_only=False):
    """

    :param filename:    filename of the spectral tiff to read.
    :return:            Tuple[spim, wavelengths, rgb, metadata], where
                        spim: spectral image cube of form [height, width, bands],
                        wavelengths: the center wavelengths of the bands,
                        rgb: a color render of the spectral image [height, width, channels] or None
                        metadata: a free-form metadata string stored in the image, or an empty string
    """
    TIFFTAG_WAVELENGTHS = 65000
    TIFFTAG_METADATA = 65111
    spim = None
    wavelengths = None
    rgb = None
    metadata = None

    first_band_page = 0
    with TiffFile(filename) as tiff:
        # The RGB image is optional, the first band image maybe on the first page:
        first_band_page = 0
        if tiff.pages[first_band_page].ndim == 3:
            rgb = tiff.pages[0].asarray()
            # Ok, the first band image is on the second page
            first_band_page = first_band_page + 1

        multiple_wavelength_lists = False
        multiple_metadata_fields = False
        for band_page in range(first_band_page, len(tiff.pages)):
            # The wavelength list is supposed to be on the first band image.
            # The older write_tiff writes it on all pages, though, so make
            # a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_WAVELENGTHS)
            tag_value = tag.value if tag else tuple()
            if tag_value:
                if wavelengths is None:
                    wavelengths = tag_value
                elif wavelengths == tag_value:
                    multiple_wavelength_lists = True
                elif wavelengths != tag_value:
                    # Well, the image is just broken then?
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing wavelength lists!')

            # The metadata string, like the wavelength list, is supposed to be
            # on the first band image. The older write_tiff wrote it on all
            # pages, too. Make a note of it.
            tag = tiff.pages[band_page].tags.get(TIFFTAG_METADATA)
    
            tag_value = tag.value if tag else ''
            if tag_value:
                if metadata is None:
                    metadata = tag_value
                elif metadata == tag_value:
                    multiple_metadata_fields = True
                elif metadata != tag_value:
                    # Well, for some reason there are multiple metadata fields
                    # with varying content. This version of the function does
                    # not care for such fancyness.
                    raise RuntimeError(f'Spectral-Tiff "{filename}" contains multiple differing metadata fields!')

        # The metadata is stored in an ASCII string. It may contain back-slashed
        # hex sequences (unicode codepoints presented as ASCII text). Convert
        # ASCII string back to bytes and decode as unicode sequence.
        if metadata:
            metadata = metadata.encode('ascii').decode('unicode-escape')
        else:
            metadata = ''

        # Some of the early images may have errorneus metadata string.
        # Attempt to fix it:
        if len(metadata) >= 2 and metadata[0] == "'" and metadata[-1] == "'":
            while metadata[0] == "'":
                metadata = metadata[1:]
            while metadata[-1] == "'":
                metadata = metadata[:-1]
            if '\\n' in metadata:
                metadata = metadata.replace('\\n', '\n')

        # Generate a fake wavelength list, if the spectral tiff has managed to
        # lose its own wavelength list.
        if not wavelengths:
            wavelengths = range(0, len(tiff.pages) - 1 if rgb is not None else len(tiff.pages))

        if multiple_wavelength_lists and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated wavelength lists!')
        if multiple_metadata_fields and not silent:
            warnings.warn(f'Spectral-Tiff "{filename}" contains duplicated metadata fields!')

        if not rgb_only:
            spim = tiff.asarray(key=range(first_band_page, len(tiff.pages)))
            spim = np.transpose(spim, (1, 2, 0))
        else:
            spim = None

        # Make sure the wavelengths are in an ascending order:
        if wavelengths[0] > wavelengths[-1]:
            spim = spim[:, :, ::-1] if spim is not None else None
            wavelengths = wavelengths[::-1]

    # Convert uint16 cube back to float32 cube
    if spim is not None and spim.dtype == 'uint16':
        spim = spim.astype('float32') / (2**16 - 1)

    return spim, np.array(wavelengths), rgb, metadata



def read_mtiff(filename):
    """
    Read a mask bitmap tiff.

    Mask bitmap tiff contains multiple pages of bitmap masks. The mask label
    is stored in tag 65001 in each page. The mask label is stored as an ASCII
    string that may contain unicode codepoints encoded as ASCII character
    sequences (see unicode-escape encoding in Python docs).

    :param filename:    filename of the mask tiff to read.
    :return:            Dict[label: str, mask: ndarray], where
                        label: the mask label
                        mask: the boolean bitmap associated with the label.
    """
    TIFFTAG_MASK_LABEL = 65001
    masks = dict()
    with TiffFile(filename) as tiff:
        for p in range(0, len(tiff.pages)):
            label_tag = tiff.pages[p].tags.get(TIFFTAG_MASK_LABEL)
            if label_tag is None:
                if p > 0:
                    print(f'** page {p}: no TIFF_MASK_LABEL tag. Ignored.')
                continue
            label = label_tag.value.encode('ascii').decode('unicode-escape')
            mask = tiff.asarray(key=p)
            masks[label] = mask > 0
    return masks


def getLabelCode(label_name, classes):
  if not label_name in classes:
      classes.append(label_name) 
  return classes.index(label_name)


def loadImagesAndMasks(root_path, IMAGE_SIZE):
    """
    INPUT:
    This function receives the RootPath that contains the folders "datasets" and "masks"
    Also, it receives the integer "IMAGE_SIZE" that will be used to resize the images into IMAGE_SIZE x IMAGE_SIZE
    
    OUTPUT:
    This function returns two Numpy Arrays: X and y
    Those corresponds to the final data that will be used in the training of the models.
    """

    REFLECTANCE_IMAGE_PATH = root_path+"Set_1_images/"
    MASK_IMAGE_PATH = root_path+"Set_1_masks/"
    classes = ["background"]
    X = []
    y=[]
    image_filenames = listdir(REFLECTANCE_IMAGE_PATH)
    for filename in image_filenames:
        spectra_image, wavelength, rgb, metadata = read_stiff(f"{REFLECTANCE_IMAGE_PATH}/{filename}")
        image_name = filename.split(".tif")[0]
        mask_image_name = f"{image_name}_masks.tif"
        image_masks_dict = read_mtiff(f"{MASK_IMAGE_PATH}/{mask_image_name}")
        masks  = np.zeros((spectra_image.shape[:2]))
        for label in image_masks_dict:
          spectra_image_mask = image_masks_dict[label].astype(int)
          label_code = getLabelCode(label,classes)
          spectra_image_mask[spectra_image_mask == 1] = label_code
          masks = np.maximum(masks,spectra_image_mask)
        y.append(resize(masks,(IMAGE_SIZE,IMAGE_SIZE)))
        X.append(resize(spectra_image,(IMAGE_SIZE,IMAGE_SIZE)))
    y = np.array(y).astype(int)
    X = np.array(X)
    return X,y, classes

def reEncodeTheMasksFormat(y, classes):
    n, h, w = y.shape
    y_reshaped = y.reshape(-1)
    return y_reshaped

def calculateSampleWeightsToBalanceTheDataSet(y_reshaped, y_train):
    class_weights = class_weight.compute_class_weight("balanced", classes = np.unique(y_reshaped), y = y_reshaped)
    class_weights = {l:c for l,c in zip(np.unique(y_reshaped), class_weights)}
    sample_weights = class_weight.compute_sample_weight(class_weights,y_train.reshape(-1,1))
    y_n, y_h, y_w = y_train.shape
    sample_weights = sample_weights.reshape(y_n,y_h,y_w)
    
    return sample_weights

def saveFileWithClassAndColorIdentifiers(classAndColorFileName, classes, class_colors):
    reshaped_classes = np.array(classes).reshape(-1,1)
    anotations = np.concatenate((reshaped_classes,class_colors),axis=1)
    np.save(classAndColorFileName,anotations)
    
def plotIoUBarGraphPerClass(IoU, classes, modelName):
    plt.rcParams["figure.figsize"] = (9,4)
    for index, iou_value in enumerate(IoU):
      plt.bar(index,iou_value)
    plt.title(modelName + " IoU_Per_Class (Test): ")
    plt.legend(classes,bbox_to_anchor=(1.05, 1),loc='upper left')
    plt.savefig('TrainingPlots/'+ modelName +'/'+'IoU_Per_Class.png', bbox_inches='tight')


def segmentAnImageWithTheModel(model, spim, IMAGE_SIZE, classes, class_colors, modelName, imageName):
    # Create the saving directory
    current_directory = os.getcwd()
    nameOfTheFolderForTheModel = modelName
    nameOfTheFolderForTheImage = imageName
    final_directory = os.path.join(current_directory,'SegmentationResults', nameOfTheFolderForTheModel, nameOfTheFolderForTheImage)
    if not os.path.exists(final_directory):
       os.makedirs(final_directory)
       
    # Perform the segmentation
    resized_image = resize(spim,(IMAGE_SIZE,IMAGE_SIZE))
    test_image = np.array([resized_image])
    test_predictions = model.predict(test_image)
    test_predictions = np.argmax(test_predictions,axis=3) # (1,128,128) size (PRETTY GOOD)
    unique_prediction_classes = np.unique(test_predictions) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    listOfRGBImagesPerClass = []
    final_segmented_image = np.zeros((IMAGE_SIZE,IMAGE_SIZE,3)) # Inizialization (128,128,3)  size with zeros
    predicted_class = dict()
    for prediction in unique_prediction_classes: # iterate inside [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      predicted_class[prediction] = classes[prediction] # Update the dictionary for laater use in the labels 
      predicted_color = class_colors[prediction] # [  0, 255, 255]
      channel0 = np.where(test_predictions[0]==prediction,predicted_color[0],0) # (128,128,1)
      channel1 = np.where(test_predictions[0]==prediction,predicted_color[1],0) # (128,128,1)
      channel2 = np.where(test_predictions[0]==prediction,predicted_color[2],0) # (128,128,1)
      currentRGBImage = np.dstack((channel0,channel1,channel2)) # (128,128,3)
      currentRGBImage = np.uint8(currentRGBImage)
    #   plt.imshow(currentRGBImage)
    #   plt.show()
      # Save the current image to the directory
      nameOfTheCurrentSubImage = str(prediction)
      # Resize from 128 to 1024
      currentRGBImageResized = Image.fromarray(currentRGBImage)
      currentRGBImageResized = currentRGBImageResized.resize((1024,1024))
      currentRGBImageResized = currentRGBImageResized.save('SegmentationResults' + '/' + nameOfTheFolderForTheModel + '/' + nameOfTheFolderForTheImage + '/' + nameOfTheCurrentSubImage +'.png')
     
      # Append this layer to the list
      listOfRGBImagesPerClass.append(currentRGBImage)
      final_segmented_image = final_segmented_image + currentRGBImage # Ronny: Sum the layers in one RGB Image
    
    final_segmented_image = np.uint8(final_segmented_image)
    # plt.imshow(final_segmented_image)
    # Save the current image to the directory
    nameOfTheCurrentSubImage = imageName
    # Resize from 128 to 1024
    final_segmented_image_resized = Image.fromarray(final_segmented_image)
    final_segmented_image_resized = final_segmented_image_resized.resize((1024,1024))
    final_segmented_image_resized = final_segmented_image_resized.save('SegmentationResults' + '/' + nameOfTheFolderForTheModel + '/' + nameOfTheFolderForTheImage + '/' + nameOfTheCurrentSubImage +'.png')
    
    # Add labels for better understanding
    handles = [
        Rectangle((0,0),1,1, color = (class_colors[index]/255)) for index in range(len(classes))
    ]
    # plt.legend(handles,predicted_class.values(),bbox_to_anchor=(1.05, 1),loc='upper left')
    # plt.title(modelName + " Segmented_Image")
    # plt.show()