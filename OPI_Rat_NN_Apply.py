#### OPI Rat CNN from "
#### Code Authors: Valerie Porter and Brent Foster
#### Version #: 1, Date: 12-22-23

#### Python Libraries: ####
import model_unet

import datetime

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers import Input, UpSampling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.utils import np_utils
from keras.utils import to_categorical


import numpy as np
from numpy import load
from numpy import ones
from numpy.random import randint
from numpy.random import uniform
from numpy import vstack
from skimage.transform import rotate, warp, resize
from skimage.io import imshow, show
from scipy.ndimage import zoom

import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
from matplotlib import pyplot

import SimpleITK as sitk

from monai.transforms import (
    AsChannelFirstd,
    LoadImaged,
    Orientationd,
    )
from monai.data import write_nifti

import glob
from os import listdir, makedirs
from os.path import isfile, exists, join

import time

#### Nifti Image Loader ####
def LoadNiftiImage(fileName):
        #### Setting up MONAI Nifti Image Loader: ####
        loader3D = LoadImaged(keys=["image"]) #Defining a name for image loader
        channel_swap = AsChannelFirstd(keys=["image"]) #Changes the last column (channel) to the first column for MONAI (want first column to be channel column)
        orientation = Orientationd(keys=["image"], axcodes="ALI")
        #Notes: Orientation options: Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).
        #(may not be accurate depending on original image orientation) 
        # Double check orientation is correct for both MRI and Label images
        
        #Loading in Images:
        data_dict = loader3D(fileName) #Loads in nifti data for one animal (pair: T2w and Label)
        data_dict = channel_swap(data_dict) #Moves color channel to the first dimension
        data_dict = orientation(data_dict) #Applies orientation settings
        img_np = data_dict 

        img_itk = sitk.ReadImage(fileName['image']) #loads in image with SimpleITK, for saving out image properties
        
        return img_np, img_itk  #MONAI loaded image, SimpleITK loaded image

#### Apply Neural Network Weights ####
    
def main():

    threshold = Threshold #Threshold probability value for which pixels will be part of the label.

    # Load the nifti images
    MRI = img_Data_Dict['image'] #Shape: (1, 128, 128, 59)
    #### Normalize Images from [0,1] ####
    npI = MRI - MRI.min()
    npI = npI / npI.max()  # Range from 0 to 1   

    #### Image Size Variable Setup ####
    #### Image Size Variable Setup ####
    originalSize = npI.shape #shape of input image volume

    IMG_HEIGHT = npI.shape[1] #image height
    IMG_WIDTH = npI.shape[2] #image width
    IMG_DEPTH = npI.shape[3] #image depth
    IMG_CHANNELS = 2 #Number of labels
  
    #### Setup of OPI Rat NN Model ####
    with tf.device('/cpu:0'):
        model = model_unet.unet(input_size=(
            IMG_HEIGHT, IMG_WIDTH, 1), numClasses=2) #loads in NN model
    model.load_weights(ModelFileName) #loads in trained weights
    outputSegmentation_np = np.zeros((IMG_DEPTH,IMG_HEIGHT,IMG_WIDTH), dtype=np.float32) #Shape: (59, 128, 128)

    #### Iterate and segment over each 2D slice: ####
    print(f"Threshold Value: {threshold}")
    print("Starting Segmentation of " + str(imgFileName))
    
    for i in range(0, IMG_DEPTH):

        tmpImg_np = npI[0, :, :, i] #Shape: (1, 128, 128, 1)

        
        print("Segmenting slice " + str(i+1) + " out of " +
              str(originalSize[3]) + " images...")  #Changed str(i) to str(i+1)
        
        #### Checks that image shape is 128x128 ####
        if tmpImg_np.shape[0] != 128:
            tmpImg_np = resize(tmpImg_np, (128, 128), order=0,
                               preserve_range=True, anti_aliasing=False)

        #### Create Label Volume #### 
        tmpImg_Channels_np = np.zeros(
            (1, tmpImg_np.shape[0], tmpImg_np.shape[1], 1), np.float32)
        tmpImg_Channels_np[0, :, :, 0] = tmpImg_np

        #### Segment the image using the trained network ####
        segmented_Img = model.predict(tmpImg_Channels_np)

        #### Combine the channels to get the label image ####
        outputSeg = np.zeros(
            (1, tmpImg_np.shape[0], tmpImg_np.shape[1], 1), np.float32)

        for j in range(1, IMG_CHANNELS):  # Skip background label
            tmp = segmented_Img[0, :, :, j]

        #### Apply a threshold to the probability values (e.g. Threshold = 0.8) ####
            tmp[tmp < threshold] = 0
            tmp[tmp != 0] = 1

            thresh_num = np.str(round(threshold*100))

        #### Multiply the image by the current image label to and add to the output discrete image  ####  
            outputSeg[0, :, :, 0] = outputSeg[0, :, :, 0] + j*tmp

        if outputSeg.shape[1] != IMG_HEIGHT:
        #### Resize to the original image size and convert back to gray scale ####
            outputSeg = resize(outputSeg[0, :, :, 0], (IMG_HEIGHT, IMG_WIDTH,
                                                       1), order=0, preserve_range=True, anti_aliasing=False)

        #### Copy the 2D image into the final output segmentation ####
        outputSegmentation_np[IMG_DEPTH-1-i, :, :] = outputSeg[0, :, :, 0]
    
#### Export the segmentation as a SimpleITK nifti image ####
    outputSegmentation = sitk.GetImageFromArray(outputSegmentation_np) #creates SimpleITK image from array
    outputSegmentation.CopyInformation(img_itk) #Copies image information from MRI image to label image

    writer = sitk.ImageFileWriter() #sets name for image writer function
    writer.SetImageIO("NiftiImageIO") #sets .nii file type
    writer.SetFileName(Output_Directory +
                       imgFileName[:-4] + "_NN_Segmented" + thresh_num + ".label.nii") #saves as .label.nii file type
    writer.Execute(outputSegmentation) #saves create label volume file
 
#### Creates Combined Image and Saves It ####
    if Export_Combined_Image == True:
        npI = MRI[0,:,:,:].transpose(2,0,1) #swaps depth 

        s = npI.shape
        s = (s[0], s[1], 2*s[2])

        combinedNp = np.zeros(s)

        # For the combine image, scale each image to be a maximum of 255
        if npI.max() != 0:
            npI = npI / npI.max() * 255

        if outputSegmentation_np.max() != 0:
            outputSegmentation_np = outputSegmentation_np / outputSegmentation_np.max() * \
                255

        combinedNp[:, :, 1:int(s[2]/2)+1] = npI[::-1,:,:] #Swapping the z-axis from back-to-front to front-to-back for the TETS Mouse Dataset
        combinedNp[:, :, int(s[2]/2):] = outputSegmentation_np
      
        tmp = sitk.GetImageFromArray(combinedNp)
        writer.SetFileName(Combined_Directory +
                           imgFileName[:-4] + "_Combined" + thresh_num[0:2] + ".label.nii") #Add label on 7-19-22
        writer.Execute(tmp)

    #Finishing Segmentation Notice:
    print("Done.")

total_start_time = time.time() #Start time for NN application of weights
if __name__ == "__main__":

    #Free up RAM in case the model definition cells were run multiple times
    K.clear_session()
        
################################### Variables to Declare ####################################
#### Directory Paths ####         
    MRI_Directory = "path/to/MRI_directory/"
    ModelFileName = "path/to/pre_trained_model_directory/pre_trained_model_filename.hdf5"
    Output_Directory = "path/to/output_directory/"
    Combined_Directory = Output_Directory + "combined_images/"

#### Threshold Values ####
    T = [0.85]   #Threshold values to iterate through. Must have [] for FOR loop.

#### Export Combined Images? ####
    Export_Combined_Image = True
    
#############################################################################################
    
    #Code creates Output Directory folder if it does not exist:
    if exists(Output_Directory) == False:
        print("\nOutput directory does not exist...")
        makedirs(Output_Directory) #Creates folder based on directory path given
        print("\nOutput directory was created...",)

#### Code creates Combined Image Directory folder if it does not exist: ####
    if exists(Combined_Directory) == False: 
            print("\nCombined image directory does not exist...")
            makedirs(Combined_Directory) #Creates folder based on directory path given
            print("\nCombined image directory was created...",)

#### Create List of MRI Filenames ####   
    MRI_FileNames = sorted(
            glob.glob(join(MRI_Directory, "*.nii"))  #Creates an alphabetically sorted list of .nii files in MRI directory
            )
#### Create Filename Dictionary for MONAI loader ####                        
    Train_FileNames_Dicts = [
            {"image": image_name}
            for image_name in MRI_FileNames #Create a dictionary for the list of MRI filenames
            ]
    
    #### Apply NN to images #### 
    for m in range(0,len(T)):
        Threshold = T[m]
        for n in range(0,len(Train_FileNames_Dicts)): 
            #### Nifti file to segment ####
            start_time = time.time() #Start time for individual scan segmentation
            imgFileName = Train_FileNames_Dicts[n]['image'][len(MRI_Directory):]  #List of Filenames ONLY of MRI data to be processed
            img_Data_Dict, img_itk = LoadNiftiImage(Train_FileNames_Dicts[n]) #End time for individual scan segmentation
                 
            main()

            #### Time duration for each scan: ####
            end_time = time.time() #End time for individual scan segmentation
            hours, rem = divmod(end_time - start_time, 3600) #Calculates number of hours and remainder
            minutes, seconds = divmod(rem, 60) #Calculates number of minutes and seconds
            print("Scan Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) #Prints individual segmentation time
                 
#### NN Appication Computation Time Calculations ####
total_end_time = time.time() #End time for NN application of weights
total_hours, total_rem = divmod(total_end_time-total_start_time, 3600) #Calculates number of hours and remainder
total_minutes, total_seconds = divmod(total_rem, 60) #Calculates number of minutes and seconds
print("Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(total_hours),int(total_minutes),total_seconds)) #Prints individual segmentations
