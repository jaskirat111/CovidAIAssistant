# Covid-19 X-Ray / CT Classification Web App
[![](https://img.shields.io/badge/python-3.7%2B-green.svg)]()
> A web app to detect covid-19 with accuracy of 93.3% from chest X-rays of patients.


> A web app to detect covid-19 with accuracy of 95% from chest CT of patients.

## Getting started in 10 minutes

- Install requirements (for gpu install tensorflow-gpu==1.15.0 instead of tensorflow-cpu==1.15.0)
- Run the script app.py
- Go to http://localhost:5000
- Done! :tada:

:point_down: Screenshot:

<p align="center">
  <img src="screenshots/homepage.png" alt="">
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

