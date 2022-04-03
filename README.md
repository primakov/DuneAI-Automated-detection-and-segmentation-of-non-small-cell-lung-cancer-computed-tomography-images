# DuneAI-Automated-detection-and-segmentation-of-non-small-cell-lung-cancer-computed-tomography-images
Original repository supporting the research paper in Nature Communications (Primakov et al. 2022)

![Method pipe-line](Fig1.png)
### **Repository structure**
#### **Usage examples:**

* Automatic segmentation script showing how to use the model for segmenting CT
* Script for calculating the RECIST measurement and tumor volume


#### **Code for reproducing the article results& figures:**

* Quantitative performance (to reproduce segmentation and detection metrics)
* Prognostic power of segmentations (to reproduce the Kaplan Meier curves for survival prediction, based on the RECIST and tumor volume calculated from automatic and manual contours)
* 'In-silico' clinical trial (to reproduce the results for time/varibility assessment and qualitative preference score)
* Comparison to a published method (to reproduce the results of comparison of proposed method to a published method)
* Software for qualitative assesment (Executable software (Windows only) for participation in the qualitative contour preference assesment) 

#### **Links for extra software:**
 
* Conversion of DICOM to nrrd was done using **Precision Medicine Toolbox**. Project GitHub Repository link: <https://github.com/primakov/precision-medicine-toolbox> 



### **Set-up**

This code was tested using Python 3.7.3 (anaconda                  2019.03), Python 3.8.5 (anaconda 2020.11), and R-studio Version 1.4.1106, on Windows 10 Enterprice x64. Hardware: Intel(R) Core i5-7200U CPU@2.5GHZ,   RAM 8,0 GB.

In order to open the files in the code repository you will need a Jupyter notebook and R-studio (for prognostic power of segmentations assessment) installed. We recommend using anaconda as it includes both Jupyter notebook and R studio. 

Before running the code it is necessary to install all the packages under requirements.txt. We recommend using a virtual environment, in order not to break existing environments and previous installations. New conda environment can be created as follows:

```
conda create -n testenv python=3.7 
```


Full guide on how to set up an environment can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

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
To run the examples in the repository we provide 3 test patients with NSCLC cases from the NSCLC-Radiomics-Interobserver1 dataset plus manual segmentations in the nrrd format. The full dataset can be found at https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics-Interobserver1. The data is distributed under the Creative Commons Attribution 3.0 Unported License https://creativecommons.org/licenses/by/3.0/ license. 

Wee, L., Aerts, H. J.L., Kalendralis, P., & Dekker, A. (2019). Data from NSCLC-Radiomics-Interobserver1 [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.cwvlpd26.

Aerts HJWL, Velazquez ER, Leijenaar RTH, Parmar C, Grossmann P, Carvalho S, Bussink J, Monshouwer R, Haibe-Kains B, Rietveld D, Hoebers F, Rietbergen MM, Leemans CR, Dekker A, Quackenbush J, Gillies RJ, Lambin P. Decoding Tumour Phenotype by Noninvasive Imaging Using a Quantitative Radiomics Approach, Nature Communications, Volume 5, Article Number 4006, June 03, 2014. DOI: http://doi.org/10.1038/ncomms5006. 

Kalendralis, P., Shi, Z., Traverso, A., Choudhury, A., Sloep, M., Zhovannik, I., Starmans, M.P., Grittner, D., Feltens, P., Monshouwer, R., Klein, S., Fijten, R., Aerts, H., Dekker, A., van Soest, J. and Wee, L. (2020). FAIR‐compliant clinical, radiomics and DICOM metadata of RIDER, Interobserver, Lung1 and Head‐Neck1 TCIA collections. Medical Physics. DOI: http://doi.org/10.1002/mp.14322.

Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7


