# DNN-CTGAN

Enhancing Intrusion Detection through Deep Learning and Generative Adversarial Network

# Table of Content
*	[Getting Started](#getting-started)
    *	[Datasets](#datasets)
    *	[Prerequisites](#prerequisites)
*	[Download and install code](#download-and-install-code)
*	[Authors](#authors)
*	[References](#references)

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Datasets
+ RDTLM implementation dataset
   + The dataset can be found in the **COVID-19 Radiography Database** directory.
   + The dataset is collected from [1].
   + To download the dataset directly from Kaggle, visit: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
+ Model validation datasets
   + The datasets can be found in the **Chest X-Ray Images (Pneumonia)** and **SARS-COV-2 Ct-Scan Dataset** directory.
   + The datasets are collected from [2] and [3].
   + Download link for **Chest X-Ray Images (Pneumonia)**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   + Download link for **SARS-COV-2 Ct-Scan Dataset**: https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset

### Prerequisites

We have tested RDTLM on MATLAB 2020a and Python on the Windows 10 operating system. You would need to install the following software before replicating this framework in your local or server machine. 

1. MATLAB 2020a or newer version

    To download the MATLAB software, visit: https://www.mathworks.com/products/matlab.html
   
2. Deep Learning Toolbox

   To download the Deep Learning Toolbox, visit: https://www.mathworks.com/products/deep-learning.html

3. Python version 3.11.6 or newer
   
    To download the Python, visit: https://www.python.org/downloads/ 
    

## Download and install code

- Retrieve the code

```
git clone https://github.com/habib-tamuk/RDTLM.git

```

- To run the program, first install all required software and toolbox. Then do the following instructions:
    - Download the [COVID-19 Radiography Database](#datasets) from [1].
    - Rename the **Normal** folder as **Healthy** and **Viral Pneumonia** folder as **Pneumonia**.
    - To measure the enhancement performance run the **Preprocessing.m** program in MATLAB.
    - To measure the clustering performance run the **Preprocessing_w_kmeans_C.m** program in MATLAB.
    - To select the value of K for K-means clustering run **Elbow Method for K selection.py** in Python.
    - Now, run the **Data_Set_Preparation.m** program in MATLAB for dataset preparation.
    - Finally to implement the RDTLM model, run the **Darknet19.m** program in MATLAB.
    - The RDTLM model used DarkNet-19 to extract the features, and mRMR was used to select the features.
    - SVM was used for the classification.
- To validate the RDTLM model with external datasets, follow the below instructions:
    - Download the [Chest X-Ray Images (Pneumonia)](#datasets) and [SARS-COV-2 Ct-Scan Dataset](#datasets) datasets from [2] and [3].
    - Run the **Data_Set_Preparation.m** program in MATLAB for dataset preparation for each dataset.
    - Now, run the **RDTLM_Validation.m** program in MATLAB with datasets [2] and [3] individually.
- To test the model with individual images, run the **Test_with_individual_image.m** program in MATLAB.

## Authors

Md Habibur Rahman, Leo Martinez III, Avdesh Mishra, Mais Nijim, Ayush Goyal and David Hicks. 

For any issue please contact Avdesh Mishra, avdesh.mishra@tamuk.edu 

## References

1. Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Al Maadeed, S., Zughaier, S.M., Khan, M.S., and Chowdhury, M.E. Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. Computers in biology and medicine, 2021, 132, 104319.
2. Kermany, D., Zhang, K., & Goldbaum, M. (2018). Labeled optical coherence tomography (oct) and chest x-ray images for classification. Mendeley data, 2(2), 651.
3. Soares, E., Angelov, P., Biaso, S., Cury, M., & Abe, D. (2023). A large multiclass dataset of CT scans for COVID-19 identification. Evolving Systems, 1-6.
