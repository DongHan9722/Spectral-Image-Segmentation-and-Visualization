"""
INDUSTRIAL PROJECT COURSE, 2021

This program take as input an original spectral data set.
After running, this program generates new folders with the PCA, RGB, MNF and ICA versions.

"""

from utils.spectral_tiffs import write_stiff, read_stiff
import numpy as np
from pysptools.noise import dnoise_int
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from pylab import *
import os
from sklearn.decomposition import PCA, FastICA
from utils.vec_spim import vec2spim, spim2vec
import shutil

def createPcaMnfRgbIcaDirectories():
    # Path of the datasets
    directory = './DataSets/OriginalDataSet/Set_1_images'
    directory_masks = './DataSets/OriginalDataSet/Set_1_masks'
    directory_rgb = './DataSets/RGBDataSet/Set_1_images'
    directory_rgb_masks = './DataSets/RGBDataSet/Set_1_masks'
    directory_mnf = './DataSets/MNFDataSet/Set_1_images'
    directory_mnf_masks = './DataSets/MNFDataSet/Set_1_masks'
    directory_pca = './DataSets/PCADataSet/Set_1_images'
    directory_pca_masks = './DataSets/PCADataSet/Set_1_masks'
    directory_ica = './DataSets/ICADataSet/Set_1_images'
    directory_ica_masks = './DataSets/ICADataSet/Set_1_masks'
    
    # Create directories
    check_rgb = os.path.isdir(directory_rgb)
    check_mnf = os.path.isdir(directory_mnf)
    check_pca = os.path.isdir(directory_pca)
    check_ica = os.path.isdir(directory_ica)
    # If folder does not exist, create it
    # RGB
    if not check_rgb:
        os.makedirs(directory_rgb)
        shutil.copytree(directory_masks, directory_rgb_masks)
    else:
        print(directory_rgb, ": RGB folder already exists.")
        return 0
    # MNF
    if not check_mnf:
        os.makedirs(directory_mnf)
        shutil.copytree(directory_masks, directory_mnf_masks)
    else:
        print(directory_mnf, ": MNF folder already exists.")
        return 0
    # PCA
    if not check_pca:
        os.makedirs(directory_pca)
        shutil.copytree(directory_masks, directory_pca_masks)
    else:
        print(directory_pca, ": PCA folder already exists.")
        return 0
    # ICA
    if not check_ica:
        os.makedirs(directory_ica)
        shutil.copytree(directory_masks, directory_ica_masks)
    else:
        print(directory_ica, ": ICA folder already exists.")
        return 0

    
    
    # Loop through the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            # Path where is saved the .tiff file
            path_read = "{}/{}".format(directory, filename)
            # Path for saving the different datasets
            path_rgb = "{}/{}".format(directory_rgb, filename)
            path_mnf = "{}/{}".format(directory_mnf, filename)
            path_pca = "{}/{}".format(directory_pca, filename)
            path_ica = "{}/{}".format(directory_ica, filename)
    
            ## DATA GENERATION
            
            # RGB dataset
            # Execute the function to read the .tiff file
            spim, wavelengths, rgb, metadata = read_stiff(path_read)
            wls_rgb = np.zeros(3).astype(np.float32)
            rgb_bands = spim[:,:,0:3]
            # Search the R(630 nm), G(532 nm) and B(465 nm) bands
            for _ in range(len(wavelengths)):
                if wavelengths[_] > 628. and wavelengths[_] < 632.:
                        wls_rgb[0] = wavelengths[_]
                        rgb_bands[:,:,0] = spim[:,:,_]
                if wavelengths[_] > 530. and wavelengths[_] < 534.:
                        wls_rgb[1] = wavelengths[_]
                        rgb_bands[:,:,1] = spim[:,:,_]
                # The wavelengths range starts from 510.1 nm. This wavelength is chosen-
                    # as blue channel
                if wavelengths[_] > 463. and wavelengths[_] < 511.:
                        wls_rgb[2] = wavelengths[_]
                        rgb_bands[:,:,2] = spim[:,:,_]
            write_stiff(path_rgb, rgb_bands, wls_rgb, rgb, metadata)
            
            # MNF dataset
            # Execute the function to read the .tiff file
            spim, wavelengths, rgb, metadata = read_stiff(path_read)
            wls_mnf = np.zeros(3).astype(np.float32) # No meaning
            mnf = dnoise_int.MNF()
            data_mnf = mnf.apply(spim)
            comp = mnf.get_components(3)
            comp_n = (comp - comp.min()) / (comp.max() - comp.min()) # values normalization
            comp_n = np.float32(comp_n)
            write_stiff(path_mnf, comp_n, wls_mnf, rgb, metadata)
            
            # PCA
            # Execute the function to read the .tiff file
            spim, wavelengths, rgb, metadata = read_stiff(path_read)
            wls_pca = np.zeros(3).astype(np.float32) # No meaning
            spim, wavelengths, rgb, metadata = read_stiff(path_read)
            # Vectorize the spectral image cube into a collection of spectrum vectors
                # by reshaping the cube:
            cube = spim
            vector = spim2vec(cube)
            pca_object = PCA()
            pca_object.fit(vector.T)
            svd_projected_vector = pca_object.transform(vector.T)
            svd_projected_cube = vec2spim(svd_projected_vector.T, cube.shape)
            im = svd_projected_cube[:,:,0:3]
            im_n = (im - im.min()) / (im.max() - im.min()) # values normalization
            write_stiff(path_pca, im_n, wls_pca, rgb, metadata)
            
            # ICA. Reference: "Band Selection Using Independent Component Analysis for Hyperspectral Image Processing", Hongtao Du, et al. 2003
            # Execute the function to read the .tiff file
            spim, wavelengths, rgb, metadata = read_stiff(path_read)
            # Vectorize the spectral image cube into a collection of spectrum vectors
                # by reshaping the cube:
            cube = spim
            vector = spim2vec(cube)
            ica = FastICA(n_components=int(38))
            ica_s = ica.fit_transform(vector)
            M = ica.mixing_
            n_bands = 3
            if M.shape[1] != np.linalg.matrix_rank(M):
                print('No inverse')
            else:
                # pseudo-inverse computation
                W = np.linalg.pinv(M)    # W. transpose(X)=transpose(S_)
                assert np.allclose(M, np.dot(M, np.dot(W, M)))  # check the pseudo-inverse matrix
            B_W = np.sum(np.absolute(W),axis=0)   # weight per band
            sortB_W = np.argsort(B_W)   # extract indexes
            bands = sortB_W[-int(n_bands):]
            # Save the images and wavelengths
            wls_ica = np.zeros(3).astype(np.float32)
            ica_bands = spim[:,:,0:3]
            for _ in range(len(bands)):
                wls_ica[_] = wavelengths[bands[_]]
                ica_bands[:,:,_] = spim[:,:,bands[_]]
            write_stiff(path_ica, ica_bands, wls_ica, rgb, metadata)
    
        else:
            continue







