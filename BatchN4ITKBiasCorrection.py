## 7-8-22 Chaudhari Lab, UC Davis, Davis, CA
## by Valerie Porter

#To Run Code:
# in Windows command console: "path\to\Slicer.exe" --python-script "path\to\BatchN4ITKBiasCorrection.py"
#Example: "C:\Users\Valerie\AppData\Local\NA-MIC\Slicer 5.0.2\Slicer.exe" --python-script "C:\Users\Valerie\Desktop\Batch Codes for 3D Slicer\BatchN4ITKBiasCorrection.py"

import time
import os

def ApplyN4ITK(ImageFilename, OutputImageFilename, ind, MaskFilename="", OutputBiasFieldFilename=""):
    [ success, MRIVolumeNode ] = slicer.util.loadVolume(ImageFilename, returnNode=True) #Loads in reference volume
    if MaskFilename != None:
        [ success, MaskSegNode ] = slicer.util.loadSegmentation(MaskFilename, returnNode=True) #Loads in atlas label as a volume
    
    OutputImageNode = slicer.vtkMRMLScalarVolumeNode() #Setting variable as a node within Slicer for saving out the output volume.
    slicer.mrmlScene.AddNode(OutputImageNode) #Adding node to scene in slicer

    if OutputBiasFieldFilename != None:
        OutputBiasFieldNode = slicer.vtkMRMLScalarVolumeNode() #Setting variable as a node within Slicer for saving out the output volume.
        slicer.mrmlScene.AddNode(OutputBiasFieldNode) #Adding node to scene in slicer

    Nodename1 = 'OutputVolume_{:02d}'.format(ind) #New node name
    OutputImageNode.CreateDefaultDisplayNodes() #Create a display node in the Slicer scene
    OutputImageNode.SetName(Nodename1) #Sets name to new node name
    
    Nodename2 = 'OutputBiasField_{:02d}'.format(ind) #New node name
    OutputImageNode.CreateDefaultDisplayNodes() #Create a display node in the Slicer scene
    OutputImageNode.SetName(Nodename2) #Sets name to new node name

    if MaskFilename != "":
        if OutputBiasFieldFilename != "":
            #Parameter setup for N4ITK Bias Field Correction:
            parameters = {'inputImageName':MRIVolumeNode, 'maskImageName':MaskSegNode, #parameters for the N4ITK Bias Correction
                          'outputImageName':OutputImageNode, 'outputBiasFieldName':OutputBiasFieldNode,
                          'initialMeshResolution':[1,1,1], 'splineDistance':0, 'bfFWHM':0,
                          'numberOfIterations':[50,40,30], 'convergenceThreshold':0.0001,
                          'bsplineOrder':3, 'shrinkFactor':4,'weightImageName':None,
                          'wienerFilterNoise':0,'nHistogramBins':0}
            print(parameters)
        elif OutputBiasFieldFilename == "":
            #Parameter setup for N4ITK Bias Field Correction:
            parameters = {'inputImageName':MRIVolumeNode, 'maskImageName':MaskSegNode, #parameters for the N4ITK Bias Correction
                          'outputImageName':OutputImageNode,
                          'initialMeshResolution':[1,1,1], 'splineDistance':0, 'bfFWHM':0,
                          'numberOfIterations':[50,40,30], 'convergenceThreshold':0.0001,
                          'bsplineOrder':3, 'shrinkFactor':4,'weightImageName':None,
                          'wienerFilterNoise':0,'nHistogramBins':0}
            print(parameters)
            
    elif MaskFilename == "":
        if OutputBiasFieldFilename != "":
            #Parameter setup for N4ITK Bias Field Correction:
            parameters = {'inputImageName':MRIVolumeNode, #parameters for the N4ITK Bias Correction
                          'outputImageName':OutputImageNode, 'outputBiasFieldName':OutputBiasFieldNode} #,
                          #'initialMeshResolution':[1,1,1], 'splineDistance':0, 'bfFWHM':0,
                          #'numberOfIterations':[50,40,30], 'convergenceThreshold':0.0001,
                          #'bsplineOrder':3, 'shrinkFactor':4,'weightImageName':None,
                          #'wienerFilterNoise':0,'nHistogramBins':0}
            print(parameters)
            
        elif OutputBiasFieldFilename == "":
            #Parameter setup for N4ITK Bias Field Correction:
            parameters = {'inputImageName':MRIVolumeNode, #parameters for the N4ITK Bias Correction
                          'outputImageName':OutputImageNode,
                          'initialMeshResolution':[1,1,1], 'splineDistance':0, 'bfFWHM':0,
                          'numberOfIterations':[50,40,30], 'convergenceThreshold':0.0001,
                          'bsplineOrder':3, 'shrinkFactor':4,'weightImageName':None,
                          'wienerFilterNoise':0,'nHistogramBins':0}
            print(parameters)

    slicer.cli.runSync(slicer.modules.n4itkbiasfieldcorrection, None, parameters) #runs Resample Image (BRAINS)

    #Create and Save Output Label Volume Node:
    slicer.util.saveNode(OutputImageNode, OutputImageFilename) #Saves transformed atlas label in the corrected data space
    slicer.util.saveNode(OutputBiasFieldNode, OutputBiasFieldFilename) #Saves transformed atlas label in the corrected data space
    slicer.mrmlScene.Clear(0) #Clears the scene of data

    
def BatchApplyN4ITK():    
   if MaskAvailable == True:
       for ind,dir in enumerate(os.listdir(path2images)): #creates of list of files in the directory
           if dir.endswith(".nii"): #Only uses .nii files, can change if using different file type.
               ImageFilename = os.path.join(path2images, dir) 
               print("Image = ", ImageFilename, end='\n') #Prints the path of Transform matrix file       

               for ind2,dir2 in enumerate(os.listdir(path2masks)): #Creates of list of files in the directory
                   if dir[0:10] == dir2[0:10]: #Matches the animal ID of the transform matrix to it's moving image label.
                       MaskFilename = os.path.join(path2masks, dir2) #Prints the path of moving label image
                       print('Mask Image Filename = ', MaskFilename, end='\n') #Prints the path of moving label file

               OutputImageFilename = os.path.join(path2output,dir[:-4]+'_BC.nii') #Path Location for saving Output Image
               print('Output Filename =', OutputImageFilename, end='\n')
               
               if BiasFieldImage == True:
                   OutputBiasFieldFilename = os.path.join(path2output,dir[:-4]+'_BF.nii') #Path Location for saving Output Image
                   print('Output Bias Field Filename =', OutputBiasFieldFilename, end='\n')

                   start_time = time.time() #Start time of individual registration
                   ApplyN4ITK(ImageFilename, OutputImageFilename, ind, MaskFilename=maskFilename, OutputBiasFieldFilename=None) #Starts batch apply transform, using function above.

               elif BiasFieldImage == False:

                   start_time = time.time() #Start time of individual registration
                   ApplyN4ITK(ImageFilename, OutputImageFilename, ind, MaskFilename=maskFilename) #Starts batch apply transform, using function above.

               end_time = time.time()#End time of individual registration
               hours, rem = divmod(end_time - start_time, 3600) #Calculates hours and remainder for minutes and seconds
               minutes, seconds = divmod(rem, 60) #Calculates minutes and seconds
               print("Registration Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) #Prints out time of individual registration


   elif MaskAvailable == False:
       for ind,dir in enumerate(os.listdir(path2images)): #creates of list of files in the directory
           if dir.endswith(".nii"): #Only uses .nii files, can change if using different file type.
               ImageFilename = os.path.join(path2images, dir) 
               print("Image Filename= ", ImageFilename, end='\n') #Prints the path of Transform matrix file       

               OutputImageFilename = os.path.join(path2output,dir[:-4]+'_BC.nii') #Path Location for saving Output Image
               print('Output Filename =', OutputImageFilename, end='\n')

               if BiasFieldImage == True:
                   OutputBiasFieldFilename = os.path.join(path2output,dir[:-4]+'_BF.nii') #Path Location for saving Output Image
                   print('Output Bias Field Filename =', OutputBiasFieldFilename, end='\n')

                   start_time = time.time() #Start time of individual registration
                   ApplyN4ITK(ImageFilename, OutputImageFilename, ind, OutputBiasFieldFilename=OutputBiasFieldFilename) #Starts batch apply transform, using function above.

               elif BiasFieldImage == False:

                   start_time = time.time() #Start time of individual registration
                   ApplyN4ITK(ImageFilename, OutputImageFilename, ind) #Starts batch apply transform, using function above.
               
               end_time = time.time()#End time of individual registration
               hours, rem = divmod(end_time - start_time, 3600) #Calculates hours and remainder for minutes and seconds
               minutes, seconds = divmod(rem, 60) #Calculates minutes and seconds
               print("Registration Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) #Prints out time of individual registration
       
total_start_time = time.time() #Start time of running code
#### MUST Declare Variables Below #####

#Declare MRI Image(s) Path:
path2images = 'path/to/MRI_directory'

#Declare Mask Image(s) Path: if no masks are needed, then declare MaskAvailable variable as False
path2masks = 'path/to/mask_directory'

#Declare Save/Output Path:
path2output = 'path/to/output_directory'

MaskAvailable = False #Parameter for whether a mask is available for bias correction
BiasFieldImage = True #Parameter for whether you would like to save the output bias field calculated.

##################################################################################################

BatchApplyN4ITK() #Calculates minutes and seconds

total_end_time = time.time() #End time of running code
total_hours, total_rem = divmod(total_end_time-total_start_time, 3600) #Calculates hours and remainder for minutes and seconds
total_minutes, total_seconds = divmod(total_rem, 60) #Calculates minutes and seconds
print("Total Time Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(total_hours),int(total_minutes),total_seconds)) #Prints total time of registration process

# Notes:
# Copy the output and save in a text file for future reference.
# If you see this warning: loadNodeFromFile `returnNode` argument is deprecated. Loaded node is now returned directly if `returnNode` is not specified.
# Do not worry. The code is running properly. If you remove returnNode from lines 19 and 20, the code will not work. So, for now, ignore this warning.
