# Better Health Hackathon on CodeforCovid19 (Finalist)

# Covid-19 Image Diagnosis Using Artfical Intelligence
The outbreak of coronavirus disease 2019 (COVID-19) has been declared a public health emergency of international concern. Epidemic studies have been well described clinical features of patients with COVID-19. The abrupt rise in the Coronavirus cases after breakout in China in the month of January, there has been a shortage of rapid and highly sensitive reverse transcriptase polymerase chain reaction (RT-PCR) testing kits for the diagnosis of coronavirus disease 2019 (COVID-19). Therefore, there is a crucial need of tools to assist the community investigating the diagnosis of patients with COVID-19 using Articial Intelligence. We have used available public medical data verified by radiologists to develop an application to initially process data in a meaningful way and create a useful knowledge that can be exploited afterwards to make prediction on patient clinical outcome. Also, the scarcity of COVID dataset is handled by using various Data augmentation techniques, the X-Ray and CT images are preprocessed to improve quality of image. Our application is a web application used to detect COVID-19 from either chest X-ray images or chest CT images. Our application can be used to increase productivity for Health Professionals by automaticaaly detecting Covid-19 infection using fast and reliable AI screening, reduce time and cost for the patients and provide assistance to overcome the problem of a lack of specialized physicians in remote villages. By using our application, the user will be able to detect AI prediction with confidence scores after uploading the Chest X-ray or CT image. The prediction results for Chest X-Ray or Chest CT Scan Image could be across three possibilities: Normal, Pneumonia or COVID-19. Furthermore, Activation Map of Chest X-Ray as well as Chest CT can be viewed by the user for observing important features reflecting the portion of chest having abnormalities like consolidation or ground glass opacities. We applied a high-quality Deep Convolutional Neural Network model (DNN) in our web application with an overall accuracy of 93.3% , COVID-19 sensitivity of 96% and COVID-19 positive prediction value (PPV) of 96% on test dataset of Chest X-Ray which is evaluated as state of the art performance. Similarly, the model was trained seperately for Covid-19 CT Datset and acheived an accuracy of 99.1% , COVID-19 sensitivity of 97.3% and COVID-19 positive prediction value (PPV) of 99.7%. The uniqueness of our project is that the propesed model contains ResNet architecture with lightweight PEPX design patterns & selective long-range connectivity pretrained on imagenet data & then trained distinctively on Chest X-Ray & CT Scan Datasets with Hyperparameter optimization & Data Augmentaion to provide accurate diagnostics for multi-class classification (COVID vs. Normal vs. Pneumonia).

## PPT Link

https://docs.google.com/presentation/d/1X04eVxvHL3xGZ4_EUbhtBY4Xal3BBAHNWRg5A4Gvad4/edit?usp=sharing

## Article Link

https://medium.com/@jaskirat_singh/social-distance-monitoring-and-face-mask-detection-ai-system-for-covid-19-6044073896d8

## Video Link

https://youtu.be/2qXDVZhai8U

## How to use Flask App
<ul>
  <li>Download repo, change to directory of Web App, go to command prompt and run <b>pip install -r requirements.txt</b></li>
  <li>On command prompt, run <b>python app.py</b></li>
  <li>Open your web browser and go to <b>127.0.0.1:5000</b> to access the Flask App</li>
</ul>

# Covid-19 X-Ray / CT Classification Web App
[![](https://img.shields.io/badge/python-3.7%2B-green.svg)]()
> A Web Application to detect signs of COVID-19 presence from Chest X-Rays and Chest CTs images using Deep Learning.


## Getting started in 10 minutes

- Install requirements (for gpu install tensorflow-gpu==1.15.0 instead of tensorflow-cpu==1.15.0)
- Run the script app.py
- Go to http://localhost:5000
- Done! :tada:

:point_down: Screenshot:

<p align="center">
  <img src="Web App/screenshots/homepage (2).png" alt="" width="90%" height="50%">
</p>

For more screenshots, please visit the <b>screenshots folder</b> of my repo, or <a href="https://github.com/jaskirat111/Covid-AI-Assistant/blob/master/Web App/screenshots">click here</a>

## Local Installation

It's easy to install and run it on your computer.

```shell
# 1. First, clone the repo
$ git clone https://github.com/jaskirat111/Covid-AI-Assistant
$ cd Covid-AI-Assistant/Web App

# 2. Install Python packages
$ pip install -r requirements.txt

# 3. Run!
$ python app.py
```

Open http://localhost:5000 and have fun. :smiley:

# COVID-19 X-Ray Dataset

**Dataset contains over 14000 Chest Xray images containing 490 COVID-19 train samples.**\

The current COVID-19 X-Ray dataset is constructed by the following open source chest radiography datasets:

* Cohen, J. P., Morrison, P. & Dao, L. COVID-19 image data collection. arXiv 2003.11597 (2020). https://github.com/ieee8023/covid-chestxray-dataset
* Chung, A. Figure 1 COVID-19 chest x-ray data initiative. https://github.com/agchung/Figure1-COVID-chestxray-dataset (2020)
* Chung, A. Actualmed COVID-19 chest x-ray data initiative. https://github.com/agchung/Actualmed-COVID-chestxray-dataset (2020).
* of North America, R. S. COVID-19 radiography database. https://www.kaggle.com/tawsifurrahman/covid19-radiography-database (2019).
* of North America, R. S. RSNA pneumonia detection challenge. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data (2019).


<!--We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Actualmed, M.E.H. Chowdhury et al., Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.-->

## COVID-19 X-Ray data distribution

Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7982  |    5459   |   490    | 14053 |
|  test |   885  |     594   |   100    |  1579 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  7966  |    5444   |    320   |  13730 |
|  test |   100  |      98   |     74   |    272 |


# COVID-19 CT Dataset

**CT Dataset contains over 100000 Chest CT images containing 12520 COVID-19 train samples.**\

* I constructed the Chest CT dataset from publicly available data provided by the China National Center for Bioinformation (CNCB). Kang Zhang, Xiaohong Liu, Jun Shen, et al. Jianxing He, Tianxin Lin, Weimin Li, Guangyu Wang. (2020). Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements and Prognosis of COVID-19 Pneumonia Using Computed Tomography. Cell, DOI: 10.1016/j.cell.2020.04.045 (http://ncov-ai.big.ac.cn/download?)


# Covid-19 CT Data Distribution
<!---
--->
Chest CT image distribution

|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  27201 |   22061   |   12520  | 61782 |
|   val |   9107 |    7400   |    4529  | 21036 |
|  test |   9450 |    7395   |    4346  | 21191 |

Patient distribution

|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |   144  |     420   |    300   |   864  |
|   val |    47  |     190   |     95   |   332  |
|  test |    52  |     125   |    116   |   293  |


## Results
These are the final results for the AI models.

### Covid19 X-Ray Model (100 COVID-19 Test Images)
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">95.0</td>
    <td class="tg-c3ow">89.0</td>
    <td class="tg-c3ow">96.0</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">90.5</td>
    <td class="tg-c3ow">93.7</td>
    <td class="tg-c3ow">96.0</td>
  </tr>
</table></div>

### Covid19 CT Scan Model

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Sensitivity (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">100</td>
    <td class="tg-c3ow">99.0</td>
    <td class="tg-c3ow">97.3</td>
  </tr>
</table></div>

<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="3">Positive Predictive Value (%)</th>
  </tr>
  <tr>
    <td class="tg-7btt">Normal</td>
    <td class="tg-7btt">Pneumonia</td>
    <td class="tg-7btt">COVID-19</td>
  </tr>
  <tr>
    <td class="tg-c3ow">99.4</td>
    <td class="tg-c3ow">98.4</td>
    <td class="tg-c3ow">99.7</td>
  </tr>
</table></div>


## Motivation

With shortages and delays in PCR tests, chest X-Rays and CTs have become one of the fastest and most affordable ways for doctors to triage patients. In many hospitals, patients often have to wait six hours or more for a specialist to look at their X-Rays or CTs. If an emergency room doctor could get an initial reading from an AI-based tool, it could dramatically shrink that wait time. Before the pandemic, health-care AI was already a booming area of research. Deep learning, in particular, has demonstrated impressive results for analyzing medical images to identify diseases like breast and lung cancer or glaucoma at least as accurately as human specialists.


## Demo

A live demo of the web application is currently running here: https://covid19-diagnosis-ai.herokuapp.com/

## How to use

**Step 1**: Open the application via a web browser and click on Detect Covid.

<p align="center">
    <img src="Web App/screenshots/homepage (2).png" alt="" width="90%" height="50%">
</p>

**Step 2**: Select whether you want to Upload the Chest X-Ray or Chest CT image for prediction. For testing purposes, you can use the sample images that we have provided in the sample folder.

<p align="center">
    <img src="Web App/screenshots/upload-image (2).png" alt="" width="90%" height="50%">
</p>


**Step 3a**: After uploading Chest X-Ray image, the Result page would show the input image and the corresponding Activation Map of the X-Ray and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="Web App/screenshots/chest-ray-result (2).png" alt="" width="90%" height="50%">
</p>



**Step 3b**: If Chest CT Image is uploaded , the Result page would show the input CT image and the corresponding Activation Map of the CT and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="Web App/screenshots/127.0.0.1_5000_uploaded_ct (1).png" alt="" width="90%" height="50%">
</p>


# Authors
## Jaskirat Singh
<ul>
<li>Github:https://github.com/jaskirat111</li>
<li>LinkedIn:https://www.linkedin.com/in/jaskirat409/</li>
</ul>


