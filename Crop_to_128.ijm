//Batch Crop and Scale (280x200) NIFTY images to (128x128) NIFTY
//By Valerie Porter Chaudhari Lab, UC Davis 7-8-2021

//Variables:
//@inputDirectory: This is the folder that contains all 280x200 MRI T2w Images
//@outputDirectory: This is the folder where you want to save all of your 
//                  cropped and scaled 128x128 images
//@outputFile: This is the variable that designates the folder location and 
//			   filename for the output file.  It uses the original file name  
//			   w/o ".nii" and adds "_crop128.nii" to the end of it.

run("Clear Results"); // clear the results table of any previous measurements

// The next line prevents ImageJ from showing the processing steps during 
// processing of a large number of images, speeding up the macro
setBatchMode(true); //Enables BatchMode

// Show the user a dialog to select a directory of images
inputDirectory = getDirectory("Choose Input Directory of Images");
outputDirectory = getDirectory("Choose Output Directory");

// Get the list of files from that directory
// NOTE: if there are non-image files in this directory, it may cause the macro to crash
fileList = getFileList(inputDirectory);

//Run actual cropping code...
print("Files that were cropped:"); //This is for the log output file that will be saved.
for (i = 0; i < fileList.length; i++)
{
	if (endsWith(fileList[i],".nii"))
		{
    		Cropto128(inputDirectory + fileList[i]); //Cropping function (see below)
    		outputFile = outputDirectory + substring(fileList[i],0,indexOf(fileList[i], ".nii")) + "_crop128.nii"; //New Filename and Folder Location
    		print(outputFile); //Adds name of output file to the log window (for record keeping, optional)
    		selectWindow("1"); //Cropping function sets the window name to 1.
    		run("NIfTI-1", "save=[" + outputFile +"]"); //Saves the output file with the new filename
    		close("*"); //closes all windows so that the appropriate window will be selected (see line 28).
		}
}

setBatchMode(false); // Now disable BatchMode since we are finished
selectWindow("Log"); //Cropping function sets the window name to 1.
saveAs("Text", ""+outputDirectory+"CropLog.txt");
run("Close");

//Cropping function: Crops image to 200x200, and then down scales to 128x128
function Cropto128(ImageFile)
{
	open(ImageFile); //Opens image file (MUST BE 280x200!)
	makeRectangle(40, 0, 200, 200); //Makes a 200x200 pixel ROI that is centered in the x-direction
	roiManager("Add"); //Adds ROI to roi manager
	run("Crop"); //Crops 280x200 image to a 200x200 image
	//This down scales the 200x200 image to 128x128 pixels:
	run("Scale...", "x=- y=- z=1.0 width=128 height=128 interpolation=Bicubic average process create title=1");
	
	
}
