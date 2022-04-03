##Automatic segmentation usage example

Since GitHub has a limitation of 100Mb for individual files, the **model weights** mdf5 file is packed in two zip archives.
Before running the script please unpack the model weights using [7zip](https://www.7-zip.org/download.html) or similiar software.
To run the example, please open the **Automatic batch segmentation.ipynb** in the Jupyter Notebook (python 3).

Test data for generating segmentations is located in "DuneAI-Au...\Software for qualitative assesment\test_data" the data shoud be automatically discovered by the script.

Input: path for folder with patient's nrrd files.

Output: folder with patient's id containing image.nrrd and DL_mask.nrrd files.

Estimated segmentation time per patient:
* with GPU(RTX2080TI): 2-3 sec
* with CPU(Core i5-7200U ): 170 -180 sec

Estimated processing time per patient depends on the multiple parameters such as: CPU/GPU usage, Hardware (HDD/SSD),and length of the CT scan (whole body scan CT/ thorax CT) 
The estimated processing time per pat. range is: 25 sec - 280 sec.

