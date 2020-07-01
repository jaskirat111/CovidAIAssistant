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
  <img src="screenshots/homepage (2).png" alt="">
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

**Dataset contains over 14000 Chest Xray images containing 473 COVID-19 train samples. Test dataset remains the same for consistency.**\

The current COVID-19 X-Ray dataset is constructed by the following open source chest radiography datasets:
* https://github.com/ieee8023/covid-chestxray-dataset
* https://github.com/agchung/Figure1-COVID-chestxray-dataset
* https://github.com/agchung/Actualmed-COVID-chestxray-dataset
* https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
* https://www.kaggle.com/c/rsna-pneumonia-detection-challenge (which came from: https://nihcc.app.box.com/v/ChestXray-NIHCC)

<!--We especially thank the Radiological Society of North America, National Institutes of Health, Figure1, Actualmed, M.E.H. Chowdhury et al., Dr. Joseph Paul Cohen and the team at MILA involved in the COVID-19 image data collection project for making data available to the global community.-->

## COVID-19 X-Ray data distribution

Chest radiography images distribution
|  Type | Normal | Pneumonia | COVID-19 | Total |
|:-----:|:------:|:---------:|:--------:|:-----:|
| train |  7966  |    5459   |   473    | 13898 |
|  test |   100  |     100   |   100    |   300 |

Patients distribution
|  Type | Normal | Pneumonia | COVID-19 |  Total |
|:-----:|:------:|:---------:|:--------:|:------:|
| train |  7966  |    5444   |    320   |  13730 |
|  test |   100  |      98   |     74   |    272 |


# COVID-19 CT Dataset
**Dataset contains 349 COVID-19 CT images from 216 patients and 397 non-COVID-19 CTs.**\

The current COVID-19 CT dataset is constructed by the following open source chest radiography datasets:
* https://github.com/UCSD-AI4H/COVID-CT

# Covid-19 CT Data Distribution
<!---
--->
Images distribution
|  Type | NonCOVID-19 | COVID-19 |  Total |
|:-----:|:-----------:|:--------:|:------:|
| train |      234    |    191   |   425  |
|  val  |       58    |     60   |   118  |
|  test |      105    |     98   |   203  |

Patients distribution
|  Type |    NonCOVID-19   | COVID-19 |  Total |
|:-----:|:----------------:|:--------:|:------:|
| train |        105       |  1-130   |   235  |
|  val  |         24       | 131-162  |    56  |
|  test |         42       | 163-216  |    96   |

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

### Covid19 CT Model

F1: 0.85

Accuracy: 0.86

AUC: 0.94


## Motivation

With shortages and delays in PCR tests, chest X-Rays and CTs have become one of the fastest and most affordable ways for doctors to triage patients. In many hospitals, patients often have to wait six hours or more for a specialist to look at their X-Rays or CTs. If an emergency room doctor could get an initial reading from an AI-based tool, it could dramatically shrink that wait time. Before the pandemic, health-care AI was already a booming area of research. Deep learning, in particular, has demonstrated impressive results for analyzing medical images to identify diseases like breast and lung cancer or glaucoma at least as accurately as human specialists.

## Acknowledgements

We would like to thank deeply the team behind [COVID-Net Open Source Initiative](https://github.com/lindawangg/COVID-Net) and Self-trans Open Source Initiative (https://github.com/UCSD-AI4H/COVID-CT) . Our project is an attempt to incorporate COVID-Net as well as Self-Trans model into the heart of a web-based application that could be used by health care providers as a supportive tool on the examination process and patient triage.

## Demo

A live demo of the web application is currently running here: https://covid-ai-assistant.herokuapp.com/

## How to use

**Step 1**: Open the application via a web browser and click on Detect Covid.

<p align="center">
    <img src="screenshots/homepage (2).png" alt="" width="90%">
</p>

**Step 2**: Select whether you want to Upload the Chest X-Ray or Chest CT image for prediction. For testing purposes, you can use the sample images that we have provided in the sample folder.

<p align="center">
    <img src="screenshots/upload-image (2).png" alt="" width="90%">
</p>


**Step 3a**: After uploading Chest X-Ray image, the Result page would show the input image and the corresponding Activation Map of the X-Ray and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="screenshots/chest-ray-result (2).png" alt="" width="90%">
</p>



**Step 3b**: If Chest CT Image is uploaded , the Result page would show the input CT image and finally the prediction Class label with confidence scores (usually this takes less than 3 seconds)

<p align="center">
    <img src="screenshots/ct-scan-result (2).png" alt="" width="90%">
</p>
