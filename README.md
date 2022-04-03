# DuneAI-Automated-detection-and-segmentation-of-non-small-cell-lung-cancer-computed-tomography-images
## **Repository structure**
Original repository supporting the research paper in Nature Communications (Primakov et al. 2022)
The repository is structured as follows:

#### **Usage examples:**

* Automatic segmentation script showing how to use the model for segmenting CT
* Script for calculating the RECIST measurement and tumor volume


#### **Code for reproducing the article results& figures:**

* Quantitative performance (to reproduce segmentation and detection metrics)
* Prognostic power of segmentations (to reproduce the Kaplan Meier curves for survival prediction, based on the RECIST and tumor volume calculated from automatic and manual contours)
* 'In-silico' clinical trial (to reproduce the results for time/varibility assessment and qualitative preference score)
* Comparison to a published method (to reproduce the results of comparison of proposed method to a published method)


#### **Links for extra software:**

* Software for qualitative assesment (Executable software (Windows only) for participation in the qualitative contour preference assesment)  LINK TO ZENODE
* Conversion of DICOM to nrrd was done using **Precision Medicine Toolbox**. Project GitHub Repository link: <https://github.com/primakov/precision-medicine-toolbox> 



### **Set-up**

This code was tested using Python 3.7.3 (anaconda                  2019.03), Python 3.8.5 (anaconda 2020.11), and R-studio Version 1.4.1106, on Windows 10 Enterprice x64. Hardware: Intel(R) Core i5-7200U CPU@2.5GHZ,   RAM 8,0 GB.

In order to open the files in the code repository you will need a Jupyter notebook and R-studio (for prognostic power of segmentations assessment) installed. We recommend using anaconda as it includes both Jupyter notebook and R studio. 

Before running the code it is necessary to install all the packages under requirements.txt. We recommend using a virtual environment, in order not to break existing environments and previous installations. New conda environment can be created as follows:

```
conda create -n testenv python=3.7 
```


Full guide on how to set up an environment can be found here: <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>

To activate the environment:
```
conda activate testenv
```
Once the virtual environment is created and activated you can install the dependencies, by first running:

```
conda install pip
```

and then navigate to the repository folder and run:

```
pip install -r requirements.txt
```

At this stage, you should be able to run the scripts.



To open the R scipt, you will need following packages installed in  the R-studio:
```
library(survival)

library(survminer)
```
you can install them by running the code down below in the R-studio console:
``` R
install.packages('survival')

install.packages('survminer')
```

## **Disclaimer**

The software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## **Example data**

LINK TO ZENODE

