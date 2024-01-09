# **OPI Rat Neural Network**
## Required Programs:
- Python 3.8.6
- CUDA 10.1
- 3D Slicer (see step # for installation of python libraries in 3D Slicers.)
- Imagej
## Python libraries to install:
- Tensorflow -v2.3.1
- Keras  -v2.4.3
- torch  -v2.1.1+cu118
- monai  -v0.5.3
- numpy  -v1.18.5
- pandas -v1.1.3
- scikit-image -v0.17.2
- scipy  -v1.5.3
- SimpleITK -v2.2.0.dev41
- simplejson -v3.17.2
- Matplotlib -v3.3.2
- Protobuf =v3.20.0 (do not upgrade; will cause error with newer versions)
## Files:
- BatchN4ITKBiasCorrection.py
    - Batch N4ITK Bias Correction code (in 3D Slicer)
- Crop_to_128.ijm – Center-cropping 280x200 images to 128x128 images (in Imagej)
- Model_unet.py 	
  - This is the RAT OPI U-Net discussed in paper: “Fully automated whole brain segmentation of rat MRI scans with a convolutional neural network”
  - You must have model_unet.py file in the same folder as the other code files.
- OPI_Rat_NN_Train.py 
  - Training model code
- OPI_Rat_NN_Apply.py 
  - Apply trained weights to images
- BatchSegmentationComparisonMetrics.py (in 3D Slicer)
## Input Data for neural network:
-	MRI Images (Size: 128x128x[any#ofslices])
-	Matching Labels (Size: 128x128x[any#ofslices])
Output Data for neural network:
- For OPI_Rat_NN_Train.py:
  -	Trained Model Weights (.hdr5 file)
  -	Training and Validation Accuracy and Loss data (.json file, total variables: 4)
  -	Training and Validation Accuracy and Loss Graphs (.png file, total graphs: 2)
- For OPI_Rat_NN_Apply:
  -	Neural network generated labels (‘.label.nii’) 
## Procedure:
### 1.	Install all required programs (see above).
1. For 3D Slicer: need to install python libraries in the python console
  1. After installing 3D Slicer, open python console by clicking on the python button on the far right-hand side of the tool bar (see red circle below)

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture1.png)

  3. Click inside the python console at the bottom of the window (see red arrow below)

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture2.png)

  5. Install pandas, nibabel, re in 3D Slicer by typing in the following lines of code 
```
import os
os.system('PythonSlicer -m pip install pandas')
os.system('PythonSlicer -m pip install nibabel')
os.system('PythonSlicer -m pip install re')
```
2. For Imagej: after installation, you will need to install nifty saver module
  1. Follow instructions on this website: https://imagej.net/ij/plugins/nifti.html

### 2.	Pre-processing MR Images for Training:
1. Make sure that the N4ITK MRI Bias correction module is installed in 3D Slicer. You can do this by opening 3D Slicer and clicking on the module search button (see red circle below)

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture3.png)

   Then, a window pop-up will appear and then, you can type N4ITK into the search bar.  
   Then, open “N4ITK MRI Bias correction” module. This should automatically be installed, but if this module does not appear in the list, you will need to install it.
   
3. Close 3D Slicer and open BatchN4ITKBiasCorrection.py in a python code editing software. I prefer python IDLE.
4. On line 143: edit path2images variable with path to MR image locations. I recommend that this is not your final image folder that you will use to input the images into the neural network, since there are multiple preprocessing steps to perform before we get to training the network.
5. (Optional): You can add a mask/label image for the N4ITK algorithm to only be applied with where the mask > 0. Be sure to edit “path2mask”, line 146, and replace the string with your path, and change “MaskAvailable”, line 151, to True.
   For our data, we did not use line 146 and set variable “MaskAvailable” = False . We gave “path2mask” a dummy path instead, but this may not be required.
6. On line 149: edit path2output variable with path to the processed MR image folder location.
7. (Optional): If you would like to save out the bias field image for each MR scan, you can set the “BiasFieldImage” variable to True. If you do not want to save it, set this variable to False.
8. To run this code, save and close the code and open the command console.
9. Entering this string: "path\to\Slicer.exe" --python-script "path\to\BatchN4ITKBiasCorrection.py" 
(an example of this string is shown on line 5 in the python code.

   The images will be saved to the location specified at step 2.5. Double check your data to see if the bias correction module was applied correctly. _BC will be added to the end of the filename for the processed image and _BF for the bias field images.

   Next, we will crop the MR images to 128x128. We have provided code that was used to convert a 280x200x59 MR volume to 128x128x59 for the OPI Rat dataset, see Crop_to_128.ijm. You can run this macro in Imagej. You can also edit this code for your matrix size needs.
  
   The Crop_to_128.ijm macro code will ask for an input folder and output folder by using the file explorer GUI. Once those folders are selected, the images will be cropped and downsized to 128x128. “_crop128” will be added to the end of the filename for the processed image.
### 3.	Pre-processing Label Images for Training:
1. If you use a different file type for the labels (i.e. .label.nii), edit macro so it finds the correct file type, see line 33. Change ".nii" to “.label.nii”
2. Perform cropping steps 2.10-2.11 for label images.
### 4.	Training Data File Organization (option 1):
1. Place MRI and Label Files in their own separate folders (i.e. Train Images and Train Labels) [show picture of example organization here]
2. OPI_Rat_NN_Train code will detect “.nii” for MR images and “.label.nii” for label images (see line 230)
3. Create a folder for training output (i.e. U-Net Model Output)
4. Go to step 6 Set Up of Training Parameters.
### 5.	Training Data File Organization (option 2):
1. Place MRI and Label Files in the same folder (i.e. Training Data)
2. OPI_Rat_NN_Train code will detect “.nii” for MR images and “.label.nii” for label images (see line 230). The code 
3. Create a folder for training output (i.e. U-Net Model Output)
### 6.	Set Up of Training Parameters (optional):
1. Open OPI_Rat_NN_Train.py in a python editing software (IDLE, Visual Studio Code, etc.)
2. Set path for folders on lines 563 through 565.
3. The code will read all of the “.nii” and “.label.nii” files in the specified folders, sort them from A-Z, and will cycle through that list. *Make sure that MR and Label files are in the same order or else that program will match the wrong MRI and label files together! 
4.	Starting at line 590, uncomment parameters that you would like to change. Default values are what are specified in the paper and are located on lines 92-116.
5.	Once all declared variables are set, you are ready to train.
### 7.	Running the Neural Network
1. I typically run this code through IDLE by hitting F5 or hovering over the “Run” tab and then selecting “Run module”.
   A shell window should pop up and print all of the model and parameter values, as well print training progression lines.

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture4.png)

   ***_Note: A warning may pop-up:_** 
 ``` 
  “Warning (from warnings module):File "C:\Program Files\Python38\lib\site-packages\monai\data\utils.py", line ...
warnings.warn(f"Modifying image pixdim from {pixdim} to {norm}")
UserWarning: Modifying image pixdim from [0.1953125 0.1953125 0.5       0.       ] to [ 0.1953125   0.1953125   0.5        22.77907507]”
```
   This is due to the monai image load module and will not affect your training. You can suppress this warning if you wish.
   
2. Now the neural network will train until completion of the specified number of epochs and keep track of the total training time.
The training graphs (2) and the shell window will not close automatically, so you can review the training session and save out the shell window output, if you desire.  The timer is done and will not be affected by these windows.
### 8.	Applying the trained neural network to new images
1.	Open OPI_Rat_NN_Apply.py in a python editing software (IDLE, Visual Studio Code, etc.)
2.	Set path for folders on lines 204 through 207. It runs through the image files in the same manner as the training code.
Notes: 
  - For ModelFileName: you are specifying the location of the .hdf5 weight file.
  - Combined_directory is the output directory where MR and neural network generated labels are combined along the right edge of the MR image, so that you can see both the label and mri in imagej at the same time. (See example down below)
   If you don’t want these images, got to line 213 and change Export_Combined_Image to False.
3.	Specify the threshold value on line 210. Our paper used T=0.85. You can specify more than one value., if you wish: (i.e. [0.85, 0.9, 0.95,…]). The minimum and maximum values that can be specified are 0 and 1, respectively.
4.	Once all declared variables are set, you are ready to run the remaining/non-training data through the neural network. The python shell window will print its progress through each scan specified.
### 9.	Perform Dice and Hausdorff Distance Calculations
1.	3D Slicer has a module you can install called “SlicerRT” that has a “SegmentComparison” module that’s part of it, link to documentation: https://www.slicer.org/wiki/Documentation/Nightly/Modules/SegmentComparison. See step 2a for how to install a module in 3D Slicer. You will be searching for the “SlicerRT” module. There is also great documentation on the slicer.org website and YouTube on how to install modules.
2.	Once you have SlicerRT installed, you can verify that you have the segmentationcomparison module installed by clicking on the module finder button:

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture5.png)

3.	Type “Segment Comparison” to find the module and double click on the name:

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture6.png)

4.	This is what the module should look like:

![alt text](https://github.com/ajchaudhari/OPI_Rat_NN/blob/main/images%20for%20readme/Picture7.png)

   If you see this module gui on your screen, then you have successfully installed the module! You can close 3D Slicer now. 
   
   I have created a batch code version that will call upon this module, but if you want to do it manually/practice what the input and out should be, then you can use the module directly in 3D Slicer first.
   
5.	Open BatchSegmentationComparisonMetrics.py
6.	Go down to line 95, where “MUST Declare Variable Below” is written. Declare the directory paths. path2reference is for the directory to the original label images, path2segmentation is for the directory to the neural network generated label images, and path2output is for the directory to where you want the output .csv file to be saved.

  	csvfilename is the name for the csv file that will contain all of the dice coefficients and Hausdorff distances.
  	
8.	Side Note: on line 112, the code finds the first few letters of the input filenames that contains the animal ID or the part of the filenames that are same to verify that the two label images are for the same subject.  
   You may need to change the element numbers within the brackets [first_element_num:last_element_num + 1] to match your ID system. For our case, our animal ID was 10 elements long, so we start at element 0 and go to element 11 (this function will not count the 11th element and stop at the 10th).  I recommend that you debug the code and double check that the code is finding the correct substring in the filenames before proceeding.  
9.	Now you can run the code by entering "path\to\Slicer.exe" --python-script "path\to\BatchSegmentationComparisonMetrics.py" in the windows command prompt window. See line 12 in the code file for more information.
