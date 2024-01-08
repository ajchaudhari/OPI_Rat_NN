## 4-5-23 Chaudhari Lab, UC Davis, Davis, CA
## by Valerie Porter
## Calculates Dice and Hausdorff Statistics in 3D Slicer and saves out the data in a .csv file.

##Run the following command in Python Interactor in Slicer to run the library nibabel and numpy for the script
##import os
##os.system('PythonSlicer -m pip install numpy')
##os.system('PythonSlicer -m pip install nibabel')
##os.system('PythonSlicer -m pip install re')

##To Run Code:
## in Windows command console: "path\to\Slicer.exe" --python-script "path\to\BatchSegmentationComparisonMetrics.py"
##Example: "C:\Users\Valerie\AppData\Local\NA-MIC\Slicer 5.0.2\Slicer.exe" --python-script "C:\Users\Valerie\...\BatchSegmentationComparisonMetrics.py"

## !!! Be sure to update the paths to your data/output file in the BatchSegmentationComparisonMetrics() definition section!!!

##Notes: Slicer will start up and nothing will appear on screen until the code has finished running. (I think the code just runs too fast for it properly capture the display?)
## The program should be done in less than a minute (for <= 120 images). May take longer with more segmentations.
## When the program is done, there will be a node called "Segmentation Accuracy Metrics" on the scene. Click the eye to right of that node to see the table.
## Be sure that you have the csv file that it is saving to, closed. Otherwise, it can't write to the file.

import os
import numpy as np
import nibabel as nib
import re

def SegmentationComparisonMetrics(ReferenceFilename, SegmentationFilename, ind): #Computes Hausdorff and Dice Statistics with SegmentationComparison module

    ReferenceVolumeNode = slicer.util.loadVolume(ReferenceFilename) #Loads in reference label as a volume, only used to determine the resolution/spacing parameters
    RefSpacing = np.array(ReferenceVolumeNode.GetSpacing(), dtype=object) #Spacing parameters
 
    ReferenceSegmentNode = slicer.util.loadSegmentation(ReferenceFilename) #Loads in reference label as a segmentation
    CompareSegNode = slicer.util.loadSegmentation(SegmentationFilename) #Loads in comparison label as a segmentation

    paramNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentComparisonNode") #Creates a SegmentComparisonNode used in the SegmentationComparison module
    paramNode.SetAndObserveReferenceSegmentationNode(ReferenceSegmentNode) #Sets reference segmentation
    paramNode.SetReferenceSegmentID("Segment_1") #Sets reference segmentation segment 
    paramNode.SetAndObserveCompareSegmentationNode(CompareSegNode) #Sets comparison segmentation
    paramNode.SetCompareSegmentID("Segment_1") #Sets comparison segmentation segment

    segmentComparisonLogic = slicer.modules.segmentcomparison.logic() #loads in segmentcomparisonmodule functions
    segmentComparisonLogic.ComputeDiceStatistics(paramNode) #Computes all Dice statistics
    segmentComparisonLogic.ComputeHausdorffDistances(paramNode) #Computes all Hausdorff Distance Statistics.

## Animal ID and Threshold Values: 
    AnimalID = os.path.basename(SegmentationFilename)[0:11]
    Threshold = str(int(re.search(r'\d+', os.path.basename(SegmentationFilename)[-23::]).group(0))/100) #os.path.basename(SegmentationFilename)[-22:-20]

##  Hausdorff Distances set to a single variable:
    maxHD = paramNode.GetMaximumHausdorffDistanceForBoundaryMm() #Max Hausdorff Distance
    averageHD = paramNode.GetAverageHausdorffDistanceForBoundaryMm() #Average Hausdorff Distance
    HD95 = paramNode.GetPercent95HausdorffDistanceForBoundaryMm() #95% Hausdorff Distance
    
##  Dice Statistics set to a single variable:
    DC = paramNode.GetDiceCoefficient() #Dice Coefficient
    TN_per = paramNode.GetTrueNegativesPercent() #True Negative as a percentage
    TP_per = paramNode.GetTruePositivesPercent() #True Positive as a percentage
    FN_per = paramNode.GetFalseNegativesPercent() #False Negative as a percentage
    FP_per = paramNode.GetFalsePositivesPercent() #False Positive as a percentage

    RefCenter_mm = abs(np.array(paramNode.GetReferenceCenter(), dtype=object)) #Reference segmentation center in mm^3
    RefCenter = abs(np.array(paramNode.GetReferenceCenter(), dtype=object)/RefSpacing) #Reference segmentation center in original voxel space
    RefVol = paramNode.GetReferenceVolumeCc() #Reference Segmenation Volume
    ComCenter_mm = abs(np.array(paramNode.GetCompareCenter(), dtype=object)) #Comparison segmentation center in mm^3
    ComCenter = abs(np.array(paramNode.GetCompareCenter(), dtype=object)/RefSpacing) #Comparison segmentation center in original voxel space
    ComVol = paramNode.GetCompareVolumeCc() #Comparison Segmentation Volume

##  Calculating Number of True/False Positives and Negatives:
    Ref_data = nib.load(ReferenceFilename)
    Seg_data = nib.load(SegmentationFilename)

    Ref_Img = Ref_data.get_fdata()
    Seg_Img = Seg_data.get_fdata()

    print(Ref_Img.shape)
    print(Seg_Img.shape)

    TP = np.sum(np.logical_and(Ref_Img[:,:,:,0] == 1, Seg_Img[:,:,:] == 1)) #Assumes the last dimension of the images is the color channel
    TN = np.sum(np.logical_and(Ref_Img[:,:,:,0] == 0, Seg_Img[:,:,:] == 0))
    FP = np.sum(np.logical_and(Ref_Img[:,:,:,0] == 0, Seg_Img[:,:,:] == 1))
    FN = np.sum(np.logical_and(Ref_Img[:,:,:,0] == 1, Seg_Img[:,:,:] == 0)) #removed last 0 for Seg_Img index because new nifty save function no longer includes a color channel when loaded in.

    Data = [os.path.basename(ReferenceFilename)[0:11],Threshold,
            DC,TP,TN,FP,FN,TP_per,TN_per,FP_per,FN_per,
            maxHD,averageHD,HD95,
            RefCenter_mm,ComCenter_mm,RefCenter,ComCenter,RefVol,ComVol] #Sets up data into an array

    #Notes: AD Rat = [0:10], OPI Rat + Mice = [0:11]

    return Data
    

def BatchSegmentationComparisonMetrics(): #Batch processes SegmentationComparison module across multiple segmentation pairs.
###################################### MUST Declare Variables Below ##########################################
    path2reference =  'path\to\reference_segmentation_directory'
    path2segmentation = 'path\to\unet_segmentation_directory'
    path2output = 'path\to\outputfile_directory'
    csvfilename = 'NN Accuracy Metrics.csv'
#########################################################################################################################
    DataHeaders = (["Animal ID", "Threshold","Dice", "TP", "TN", "FP", "FN", "TP(%)", "TN(%)", "FP(%)", "FN(%)",
                     "Max HD (mm)", "AVG HD (mm)", "95% HD (mm)", "Ref Center", "Seg Center",
                     "Ref Center (mm^3)", "Seg Center (mm^3)", "Ref Vol (cc)", "Seg Vol(cc)"]) #Headers for data table
    Data = [] #Initializes the variable
                     
    for ind,dir in enumerate(os.listdir(path2reference)): #creates of list of files in the directory
        
        if dir.endswith("label.nii"):
            ReferenceFilename = os.path.join(path2reference, dir) 
            print("Reference Segmentation Image = ", ReferenceFilename, end='\n') #Prints the path of Transform matrix image the file
        
            for ind2,dir2 in enumerate(os.listdir(path2segmentation)): #creates of list of files in the directory
                if dir[0:11] == dir2[0:11] and dir2.endswith(".nii"): #matches the animal ID [change numbers in square brackets for your data] of the transform matrix to it's reference image.
                    SegmentationFilename = os.path.join(path2segmentation, dir2) 
                    print('Comparison Segmentation Image = ', SegmentationFilename, end='\n') #Prints the path of reference image the file

                    print("Calculating Metrics...")
                    DataNew = SegmentationComparisonMetrics(ReferenceFilename, SegmentationFilename, ind) #Metric data calculated with SegmentationComparison function above.
                    print(DataNew[1])
                    slicer.mrmlScene.Clear(0) #Only clears nodes/images in 3D Slicer
                    Data.append(DataNew) #Appends each segmentation pair's accuracy metrics to an array.

    print("Calculations Done!")
    
    if not os.path.exists(path2output): #Creates the file path if it does not exist.
        os.makedirs(path2output)

    Outputfilename = os.path.join(path2output, csvfilename) #Path Location for saving Output Image
    print('Output Filename =', Outputfilename, end='\n') #Prints out the filename in the python interactor.


    Data = np.array(Data,dtype=object) #Changes Data array into an numpy array.
                
    resultTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "Segmentation Accuracy Metrics") #Creates a table node in 3D Slicer to save out metrics
    tableWasModified = resultTableNode.StartModify() #Allows us to modify the table node

    for i in range(0, Data.shape[1]): #Sets up the columns and column names (each accuracy metric) of the table node in 3D Slicer
        resultTableNode.AddColumn() #Adds an empty column
        resultTableNode.GetTable().GetColumn(i).SetName(DataHeaders[i]) #Sets column name
    
    for j in range(0, Data.shape[0]): #Sets up the rows and inputs data in the table node in 3D Slicer (number of rows = number of segmentation comparisons)
        resultTableNode.AddEmptyRow() #Adds an empty row
        
        for k in range(0,Data.shape[1]): 
            resultTableNode.SetCellText(j,k, str(Data[j,k])) #Transfers data from Data array into table (node)

    resultTableNode.EndModify(tableWasModified) #Signals that we are done modifying the table node

    slicer.util.saveNode(resultTableNode, Outputfilename) #saves the table node as a .csv file.
        
        
BatchSegmentationComparisonMetrics() #Runs the entire code.
