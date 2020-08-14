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
  <img src="screenshots/homepage (2).png" alt="" width="90%" height="50%">
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

* Cohen, J. P., Morrison, P. & Dao, L. COVID-19 image data collection. arXiv 2003.11597 (2020).
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
Images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  27201 |    22061  |   12520  | 61782 |
|  Val  |   9107 |     7400  |   4529   | 21036 |
|  test |   9450 |     7395  |   4346   | 21191 |

## Results
These are the final results for the AI models.

### Covid19 X-Ray Model (490 COVID-19 Test Images)
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
    <img src="screenshots/homepage (2).png" alt="" width="90%" height="50%">
</p>

**Step 2**: Select whether you want to Upload the Chest X-Ray or Chest CT image for prediction. For testing purposes, you can use the sample images that we have provided in the sample folder.

<p align="center">
    <img src="screenshots/upload-image (2).png" alt="" width="90%" height="50%">
</p>


**Step 3a**: After uploading Chest X-Ray image, the Result page would show the input image and the corresponding Activation Map of the X-Ray and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="screenshots/chest-ray-result (2).png" alt="" width="90%" height="50%">
</p>



**Step 3b**: If Chest CT Image is uploaded , the Result page would show the input CT image and the corresponding Activation Map of the CT and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="screenshots/127.0.0.1_5000_uploaded_ct (1).png" alt="" width="90%" height="50%">
</p>


