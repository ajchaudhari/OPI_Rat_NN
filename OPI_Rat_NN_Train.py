#### OPI Rat CNN from "
#### Code Authors: Valerie Porter and Brent Foster
#### Version #: 1, Date: 12-22-23

#### Python Libraries: ####                           
import tensorflow as tf
import keras
from keras import backend as K
import datetime
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

import model_unet

import pause as ps
import numpy as np
from numpy import load
from numpy import ones
from numpy.random import randint
from numpy.random import uniform

from skimage.transform import warp, rotate, resize
from skimage.io import imshow, show
from scipy.ndimage import zoom

import math

import json
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab

import torch
from monai.transforms import (
    AsChannelFirstd,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand2DElastic,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandShiftIntensity,
    Spacingd,
)
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import glob
from os import listdir, makedirs
from os.path import isfile, isdir, exists, join, split

#### Modified U-Net Training Code ####
class UNet_Segmentation(object):
    """
    docstring
    """
    #### Required Directory and Filename Inputs ####
    def __init__(self, MRI_Directory, Label_Directory, Output_Directory, OutputModelFileName, JSONFilename, OutputHistoryFileName1, OutputHistoryFileName2, OutputHistoryFileName3, OutputHistoryFileName4):
        self.MRI_Directory = MRI_Directory
        self.Label_Directory = Label_Directory
        self.Output_Directory = Output_Directory
        self.JSONFilename = JSONFilename
        self.OutputModelFileName = OutputModelFileName
        self.OutputHistoryFileName1 = OutputHistoryFileName1
        self.OutputHistoryFileName2 = OutputHistoryFileName2
        self.OutputHistoryFileName3 = OutputHistoryFileName3
        self.OutputHistoryFileName4 = OutputHistoryFileName4

        check = isdir(self.Output_Directory)

        if check == False:
            raise ValueError('Directory for saving data does not exist or has a typo.')
#### Default values for data augmentation parameters ####
        #### Set equal to 0 to skip that particular augmentation type ####
        self.TranslationX = 40 #Units = [# of pixels] 
        self.TranslationY = 40 #Units = [# of pixels]
        self.rotAngle = 45 #Units = [degrees]
        self.brightnessVal = 0.5 
        self.scaleVal = 0.3
        self.noiseMean = 1 
        self.noiseVal = 0.25 
        self.Flip = True 

        #### Other parameters ####
        self.Apply_Augmentation = True #MONAI Apply data augmentation
        self.batch_size = 10 
        self.epochs = 150 
        self.steps_per_epoch = 20
        self.lossFunction = 'categorical_crossentropy' #Loss function
        self.learning_rate = 0.0002 

       
        self.SkipZeroLabelImages = True #Only include images which have a non-zero label 
        #This can help with the training is there are many images with only the background label
        self.UseOnlyFirstFile = False #Only use the first image, for training or debugging
        self.Apply_Augmentation = True #Apply data augmentation to training images                
        self.ApplyClassWeights = True #Apply class weighting for training labels

        #### Print the parameters in the code output shell to save: ####
        print('Data Augmentation Parameters:')
        print(f"self.TranslationX = {self.TranslationX}")
        print(f"self.TranslationY = {self.TranslationY}")
        print(f"self.rotAngle = {self.rotAngle}")
        print(f"self.brightnessVal = {self.brightnessVal}")
        print(f"self.scaleVal = {self.scaleVal}")
        print(f"self.noiseMean = {self.noiseMean}")
        print(f"self.noiseVal = {self.noiseVal}")
        print(f"self.Flip = {self.Flip}")
        print(f"self.Apply_Augmentation = {self.Apply_Augmentation}")

        print('2D U-net Model Training Parameters:')
        print(f"self.batch_size= {self.batch_size}")
        print(f"self.epochs = {self.epochs}")
        print(f"self.steps_per_epoch = {self.steps_per_epoch}")
        print(f"self.UseOnlyFirstFile = {self.UseOnlyFirstFile}")
        print(f"self.lossFunction = {self.lossFunction}")
        print(f"self.learning_rate = {self.learning_rate}")
        print(f"self.SkipZeroLabelImages = {self.SkipZeroLabelImages}")

        
        #### Create List of MRI Filenames ####
        self.MRI_FileNames = sorted(
            glob.glob(join(self.MRI_Directory, "*.nii"))
            )
        #### Create List of Label Filenames ####                              
        self.Label_FileNames = sorted(
            glob.glob(join(self.Label_Directory, "*.labels.nii"))
            )
        #### Create Filename Dictionary for MONAI loader ####                              
        self.Train_FileNames_Dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(self.MRI_FileNames, self.Label_FileNames)
            ]

        #### Load the first label image to get the image size and the number of distinct segmentation labels ####
        img_ydim, img_xdim, numClasses = self.GetImageSize() #image height, image width, Number of labels
        self.img_ydim = img_ydim #image height
        self.img_xdim = img_xdim #image width
        self.numClasses = numClasses

        #### Initialize the U-Net model on the CPU scope so that the model's weights are hosted on CPU memory ####
        #### To avoid them being hosted on the GPU (which may be slower) ####
        with tf.device('/cpu:0'):
            self.model = model_unet.unet(input_size=(
                self.img_ydim, self.img_xdim, 1), numClasses=self.numClasses)

        if self.lossFunction == "jaccard":
            self.model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1),
                               loss=self.jaccard_coef,
                               metrics=[self.jaccard_coef, 'accuracy', 'jaccard'])
        else:
            self.model.compile(optimizer=Adam(lr=self.learning_rate, clipnorm=1),
                               loss=self.lossFunction,
                               metrics=[self.jaccard_coef, 'accuracy', 'categorical_accuracy'])
        self.model.summary()

    #### MONAI Data Augmentation ####
        #See https://docs.monai.io/en/latest/transforms.html for more information 
    def DataAugmentation(self, MRI, Label):
        #### Random Transformation: rotate, scale, translation ####
        randaffine = RandAffine(
            prob = 0.3, #30% chance
            rotate_range=(self.rotAngle),
            scale_range=(self.scaleVal),
            translate_range=(self.TranslationX, self.TranslationY),
            padding_mode="zeros",
            device=torch.device('cpu'), 
        )
        #### Random Flip: ####
        randflip = RandFlip(
            prob=0.5, #50% chance
            spatial_axis = randint(2)
        ) #Randomly selects a spatial axis and flip along it. 
        
        #### Random Gaussian Noise: ####
        randnoise = RandGaussianNoise(
            prob=0.1, #10% chance
            mean=self.noiseMean,
            std=self.noiseVal
        ) #Randomly adds gaussian noise to image

        randshiftintensity = RandShiftIntensity(
            prob=0.3, #30% chance
            offsets=self.brightnessVal
        ) #Randomly offsets the intensity values of the image.
        
        #### Apply MONAI Transformation: ####
        # converts both image and segmentation using bilineary interpolation mode 
        n = np.random.randint(0,100000) #This allows for both the image and label to have the same transformationspeformed on them
        
        #Flips the images 50% of the time (50/50 for LR or UD Flip) 
        randflip.set_random_state(seed=n)
        New_MRI = randflip(MRI)
        randflip.set_random_state(seed=n)
        New_Label = randflip(Label)

        #Transforms the Image with rotations, scales, and translations (see above) 
        randaffine.set_random_state(seed=n)
        New_MRI = randaffine(New_MRI, (128, 128), mode="bilinear")
        randaffine.set_random_state(seed=n)
        New_Label = randaffine(New_Label, (128, 128), mode="nearest")

        #Adds Gaussian Noise 10% of the Time to just the MRI image 
        randnoise.set_random_state(seed=n)  
        New_MRI = randnoise(New_MRI)

        #### Shifts Brightness Pixel Vaules 30% of the Time to just the T2w 
        randshiftintensity.set_random_state(seed=n)  
        New_MRI = randshiftintensity(New_MRI)
        
        return New_MRI, New_Label
        
    #### Import Previously Trained Weights ####    
    def ImportModelWeights(self, filePath):
        # Load a previously trained model (must be the same size as this model)
        self.model.load_weights(filePath)
    
    #### Load a Random MR Image and Matching Label Image ####
    def Load_Random_Image(self):
        # This function loads a random MRI (and corresponding label nifti file)
        # Then randomly selects a 2D image from the 3D volume
        # (Optional) Data augmentation is applied to the image

        FileNdx = int(
            randint(0, min(len(self.MRI_FileNames), len(self.Label_FileNames)), 1)
            )

        # Optional: Only use the first image for training or debugging
        if self.UseOnlyFirstFile == True:
            FileNdx = 0

        # Load the nifti images
        Data_Dict = self.LoadNiftiImage(
            self.Train_FileNames_Dicts[FileNdx]
            )
        MRI, Label = Data_Dict['image'], Data_Dict['label']
        New_MRI = MRI - MRI.min()
        New_MRI = New_MRI / New_MRI.max()  # Range from 0 to 1
               
        # Re-number the values of the label image to be integers from one to number of channels (such as 1 to 15)
        vals = np.unique(Label)

        for i in range(0, len(vals)):
##            if vals[i] <= len(vals) & vals[i] != 0:
##                raise("Unique value is less than the number of unique values")
            Label[Label == vals[i]] = i

        # Randomly select a 2D slice from the 3D volume
        # Optionally: Only include images which have a non-zero label
        # This can help improve training if there are many more images with only the background label then labeled images
        if self.SkipZeroLabelImages == True:
            while True:
                sliceNdx = int(randint(0, self.img_slices, 1))
                tmp = Label[:, :, :,]
                if tmp.max() != 0:
                    break
        else:
            sliceNdx = int(randint(0, self.img_slices, 1))

        MRI = MRI[:, :, :, sliceNdx]
        Label = Label[:, :, :, sliceNdx]
        
        #Apply data augmentation if set to True:
        if self.Apply_Augmentation == True:
            New_MRI, New_Label = self.DataAugmentation(MRI, Label)


        # Apply a normalization to the MRI image to have the intensity values range from 0 to 1
        New_MRI = New_MRI - New_MRI.min()
        New_MRI = New_MRI / New_MRI.max()  # Range from 0 to 1

        return New_MRI, New_Label

    #### Training Data Batch Generator ####
    def generate_train_batch(self, batch_size):
        # This function is called during model fitting and returns a set of images and labels

        while 1:

            imgs = np.zeros((batch_size, self.img_xdim,
                             self.img_ydim), np.float32)
            labels = np.zeros((batch_size, self.label_ydim,
                               self.label_ydim), np.float32)

            for i in range(0, batch_size):
                imgs[i, :, :], labels[i, :, :] = self.Load_Random_Image()

            # This part may be a bit confusing
            # We need to have a copy of the image for each class label (which is equal to the number of segmentation labels)
            # For example, suppose batch_size = 10, img_height = 128, img_width = 128, and numClasses = 2
            # Size of imgs would then be [10, 128, 128, 2] where the 4th dimension is just copies of the image

            batch_imgs = np.zeros(
                (batch_size, self.img_ydim, self.img_xdim, 1), np.float32)
            
            for i in range(0, batch_size):
                batch_imgs[i, :, :, 0] = imgs[i, :, :]

            # Similarly, we need to convert the label images to a binary class matrix which contains either 0 or 1
            batch_labels = np_utils.to_categorical(labels, self.numClasses)

            # Lower the weight of the background label
            for i in range(0, self.numClasses):
                batch_labels[:, :, :, i] = self.classWeights[i] * batch_labels[:, :, :, i]

            yield batch_imgs, batch_labels
    
    #### MONAI Nifti Image Loader ####
    def LoadNiftiImage(self, fileName):
        #Loading Nifti T2w and Label Files:
        #Setting up MONAI Nifti Images Loader:
        loader3D = LoadImaged(keys=["image", "label"]) #Defining a name for the loader function with nibablel for loading in .nii files
        channel_swap = AsChannelFirstd(keys=["image", "label"])
        orientation = Orientationd(keys=["image", "label"], axcodes="ALI") ##The default axis labels are Left (L), Right (R), Posterior (P), Anterior (A), Inferior (I), Superior (S).

        #Loading in Images:
        data_dict = loader3D(fileName) #Loads in nifti data for one animal (pair: T2w and Label)
        if data_dict['label'].shape == (128,128,59):
            label_temp = np.expand_dims(data_dict['label'],axis=3)
            data_dict['label'] = label_temp
        data_dict = channel_swap(data_dict) #Changes the last column (channel) to the first column for MONAI (want first column to be channel column)
        data_dict = orientation(data_dict) #Correctes the images from LR (x-axis) to TB (y-axis) orientation
        img_np = data_dict
        
        return img_np
        
    #### Get Image Size and Channel Information ####
    def GetImageSize(self):
        #Load the first label image to get the image sizes
        Data_Dict = self.LoadNiftiImage(self.Train_FileNames_Dicts[0])
        #Pull only image data from dictionary
        MRI, Label = Data_Dict['image'], Data_Dict['label']
        
        
        self.img_vol_shape = MRI.shape #All dimensions of the image volume (channels,x,y,z)
        self.img_num_channels = self.img_vol_shape[0] #Number of channels from image
        img_xdim = self.img_vol_shape[1] #x dimension of image
        img_ydim = self.img_vol_shape[2] #y dimension of image
        self.img_slices = self.img_vol_shape[3] #number of slices in image

        self.label_vol_shape = Label.shape #All dimensions of the label volume (channels,x,y,z)
        self.label_num_channels = self.label_vol_shape[0] #Number of channels from label
        self.label_xdim = self.label_vol_shape[1] #x dimension of label
        self.label_ydim = self.label_vol_shape[2] #y dimension of label
        self.label_slices = self.label_vol_shape[3] #number of slices in label

        # Number of output segmentation labels
        numClasses = len(np.unique(Label))
        
        # Re-number the values of the label image to be integers from one to number of channels (such as 1 to 15)
        vals = np.unique(Label)

        for i in range(0, len(vals)):  
            Label[Label == vals[i]] = i

        # Estimate the relative volumes of each label to apply a scaling to have a more equal class scaling
        self.classWeights = np.zeros(numClasses)
        for i in range(0, numClasses):
            self.classWeights[i] = len(np.argwhere(Label == i))

        self.classWeights = self.classWeights / np.sum(self.classWeights)
        self.classWeights = 1 - self.classWeights
        
        return img_ydim, img_xdim, numClasses
        
#### Jaccard Coefficient Calculations During Training (Optional) ####
    def jaccard_coef(self, y_true, y_pred): #Does not calculate the jaccard properly.
        # Calculate the Jaccard overlap coefficient between two label images

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        denominator = (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)  #Original denominator

        if denominator != 0:
            jaccard = (intersection + 1.0) / denominator  #Original jaccard calculation
            #jaccard = intersection/denominator
        else:
            jaccard = 0.

        return jaccard
#### Jaccard Coefficient Loss Function (Optional) ####
    #Can use instead of categorical accuracy
    def jaccard_coef_loss(self, y_true, y_pred):
        # Use the Jaccard coefficient as the CNN loss function
        return -1*self.jaccard_coef(y_true, y_pred)

#### Model Training Code: ####
    def Train(self):
        # Stop the training early if it has not improved the loss function in this number of epochs
        earlystopper = EarlyStopping(patience=5, verbose=1)

        gen_train = self.generate_train_batch(self.batch_size)

        history = {}

        for epoch in range(0, self.epochs):
            
            self.history = self.model.fit(gen_train, validation_data=next(gen_train), steps_per_epoch=self.steps_per_epoch,epochs=1,callbacks=[earlystopper]).history
            
            self.PlotExample(step=epoch) # Save an example segmentation figure to the disk

            self.OutputModelFileName = [OutputModelFileName[0:-5] + '_E' +int(epoch) +'.hdf5']
            
            self.model.save_weights(self.OutputModelFileName) # Save the model weights to the disk

            if epoch == 0:
                history = self.history

            else:
                for key in history.keys():
                    history[key].append(self.history[key][0])
                    
            print(f'Epochs Completed: {epoch+1} out of {self.epochs}')
                        
        self.model.save_weights(self.OutputModelFileName)

        with open(self.JSONFilename, 'w') as f:  # Python 3: open(..., 'wb')
            json.dump(history, f)
        #### Notes: Dictionary Keys in JSON File ####
        #dict_keys(['loss', 'jaccard_coef', 'accuracy', 'categorical_accuracy',
        #'val_loss', 'val_jaccard_coef', 'val_accuracy', 'val_categorical_accuracy'])

        x_start = len(history['accuracy'])
        x = np.arange(1,x_start+1)
        xlim = np.arange(0, x_start+1,10)
        xlim[0] = 1

        #### Time Duration of NN Training ####
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        #### Creating the x-axis data for the graphs of the NN training: ####
        x_end = len(history['accuracy'])
        x = np.arange(1,x_end+1)
        x1 = np.arange(1,x_end/2 + 1)
        xlim = np.arange(0, x_end+1,10)
        xlim[0] = 1

        #### Calculating the figure size, based on the number of data points: ####
        x_tick_spacing = 0.5 #in inches
        w = max([6.4, x_tick_spacing*len(xlim)]) #Width of figures below, 6.4 is the default setting for the width of plt.figure().
        h = 4.8 #in, Height of figures below, 4.8 is the default setting for plt.figure().
        #### Summarize history for accuracy: ####
        plt.figure(1, figsize=[w,h])
        plt.plot(x, history['accuracy'])
        plt.plot(x, history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(xlim)
        plt.show(block=False)
        plt.savefig(self.OutputHistoryFileName1)
                    
        # summarize history for loss:
        plt.figure(2, figsize=[w,h])
        plt.plot(x, history['loss'])
        plt.plot(x, history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.xticks(xlim)
        plt.show(block=False)
        plt.savefig(self.OutputHistoryFileName2)


    #### Plot set of original and nn-generated labels at each epoch to monitor training progress ####
    def PlotExample(self, step=0):

        # Only plot up to 5 images or else they are too small to see well
        n_samples = np.min((self.batch_size, 5))
        generator = self.generate_train_batch(n_samples)

        imgs, labels = next(generator)
        PredictedLabels = self.model.predict(imgs)

        self.jaccard_coef(labels, PredictedLabels)

        

        # Combine the channels to get the final label image
        outputSeg = np.zeros(
            (n_samples, PredictedLabels.shape[1], PredictedLabels.shape[2], 1), np.float32)
        GT_Image = np.zeros(
            (n_samples, PredictedLabels.shape[1], PredictedLabels.shape[2], 1), np.float32)

        for i in range(0, n_samples):
            for j in range(0, self.numClasses):
                outputSeg[i, :, :, 0] = outputSeg[i, :, :, 0] + \
                    j * PredictedLabels[i, :, :, j]

                GT_Image[i, :, :, 0] = GT_Image[i, :, :, 0] + \
                    j * labels[i, :, :, j]

        s = imgs.shape
        # Plot the MRI images
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(imgs[i], cmap='gray')
            else:
                plt.imshow(imgs[i, :, :, 0], cmap='gray')

        # Plot the predicted labels
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(outputSeg[i])
            else:
                plt.imshow(outputSeg[i, :, :, 0])

        # Plot ground truth label
        for i in range(n_samples):
            plt.subplot(3, n_samples, 1 + n_samples*2 + i)
            plt.axis('off')

            if len(s) == 3:
                plt.imshow(GT_Image[i])
            else:
                plt.imshow(GT_Image[i, :, :, 0])

        # save plot to file
        OutputDirectory = split(self.OutputModelFileName)
        filename1 = OutputDirectory[0] + '/Plot_%03d.png' % (step+1)
        plt.savefig(filename1)
        plt.close()

        
start_time = time.time() #Start time for NN learning
if __name__ == "__main__":
    
    # Free up RAM in case the model definition cells were run multiple times
    K.clear_session()
################################# Path Variables to Declare ###################################
#### Directory Paths ####        
    MRI_Directory = "path/to/MRI_directory/"
    Label_Directory = "path/to/label_directory/"
    Output_Directory = "path/to/output_directory/"

    #### Output model weights (.hdf5) ####
    Output_Model_FileName = Output_Directory + "weights_filename.hdf5"

    #### Output model history (.json) and training graphs (.png): ####
    JSON_Filename = Output_Directory + "history_filename.json"
    Output_Accuracy_FileName = Output_Directory + "Accuracy_Graph_filename.png" 
    Output_Loss_FileName = Output_Directory + "Loss_Graph_filename.png"
    
#############################################################################################
    #### Create Output File Folders ####
    #Code creates Output Directory folder if it does not exist:
    if exists(Output_Directory) == False:
        print("\nOutput directory does not exist...")
        makedirs(Output_Directory)
        print("\nOutput directory was created...",)

    # Initiation of U-Net file:
    U_Net = UNet_Segmentation(
        MRI_Directory, Label_Directory, Output_Directory, OutputModelFileName, JSONFilename, OutputHistoryFileName1, OutputHistoryFileName2, OutputHistoryFileName3, OutputHistoryFileName4) 
    
############################ Training Parameters to Declare (Optional) #################################   
#### Data Augmentation Parameters ####
# Set equal to 0 to skip that particular augmentation type # 
    #U_Net.TranslationX = 40
    #U_Net.TranslationY = 40
    #U_Net.rotAngle = 45 
    #U_Net.brightnessVal = 0.5 
    #U_Net.scaleVal = 0.3
    #U_Net.noiseMean = 1 #Final
    #U_Net.noiseVal = 0.25 #Final
    #U_Net.Flip = True #Changed to True from False

#### Training Parameters ####
    #U_Net.Apply_Augmentation = True
    #U_Net.batch_size = 10
    #U_Net.epochs = 150 
    #U_Net.steps_per_epoch = 20
    #U_Net.lossFunction = 'categorical_crossentropy'
    #U_Net.learning_rate = 0.0002

#### Optional: Parameters ####
    # Only use the first image, for training or debugging
    #U_Net.UseOnlyFirstFile = False
    
    # Only include images which have a non-zero label #
    #U_Net.SkipZeroLabelImages = True # This can help with the training is there are many images with only the background label

    # Use Data Augmentation #
                          
    #U_Net.Apply_Augmentation = True

    # Use class weights #
                                                      
    #U_Net.ApplyClassWeights = True
#############################################################################################
    
    U_Net.Train()
