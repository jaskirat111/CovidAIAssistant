# Covid-19 Image Diagnosis Using Artfical Intelligence
The outbreak of coronavirus disease 2019 (COVID-19) has been declared a public health emergency of international concern. Epidemic studies have been well described clinical features of patients with COVID-19. The abrupt rise in the Coronavirus cases after breakout in China in the month of January, there has been a shortage of rapid and highly sensitive reverse transcriptase polymerase chain reaction (RT-PCR) testing kits for the diagnosis of coronavirus disease 2019 (COVID-19). Therefore, there is a crucial need of tools to assist the community investigating the diagnosis of patients with COVID-19 using Articial Intelligence. We have used available public medical data verified by radiologists to develop an application to initially process data in a meaningful way and create a useful knowledge that can be exploited afterwards to make prediction on patient clinical outcome. Also, the scarcity of COVID dataset is handled by using various Data augmentation techniques, the X-Ray and CT images are preprocessed to improve quality of image. Our application is a web application used to detect COVID-19 from either chest X-ray images or chest CT images. Our application can be used to increase productivity for Health Professionals by automaticaaly detecting Covid-19 infection using fast and reliable AI screening, reduce time and cost for the patients and provide assistance to overcome the problem of a lack of specialized physicians in remote villages. By using our application, the user will be able to detect AI prediction with confidence scores after uploading the Chest X-ray or CT image. The prediction results for Chest X-Ray or Chest CT Scan Image could be across three possibilities: Normal, Pneumonia or COVID-19. Furthermore, Activation Map of Chest X-Ray as well as Chest CT can be viewed by the user for observing important features reflecting the portion of chest having abnormalities like consolidation or ground glass opacities. We applied a high-quality Deep Convolutional Neural Network model (DNN) in our web application with an overall accuracy of 93.3% , COVID-19 sensitivity of 96% and COVID-19 positive prediction value (PPV) of 96% on test dataset of Chest X-Ray which is evaluated as state of the art performance. Similarly, the model was trained seperately for Covid-19 CT Datset and acheived an accuracy of 99.1% , COVID-19 sensitivity of 97.3% and COVID-19 positive prediction value (PPV) of 99.7%. The uniqueness of our project is that the propesed model contains ResNet architecture with lightweight PEPX design patterns & selective long-range connectivity pretrained on imagenet data & then trained distinctively on Chest X-Ray & CT Scan Datasets with Hyperparameter optimization & Data Augmentaion to provide accurate diagnostics for multi-class classification (COVID vs. Normal vs. Pneumonia).

## How to use Flask App
<ul>
  <li>Download repo, change to directory of Web App, go to command prompt and run <b>pip install -r requirements.txt</b></li>
  <li>On command prompt, run <b>python app.py</b></li>
  <li>Open your web browser and go to <b>127.0.0.1:5000</b> to access the Flask App</li>
</ul>


# Authors
## Jaskirat Singh
<ul>
<li>Github:https://github.com/jaskirat111</li>
<li>LinkedIn:https://www.linkedin.com/in/jaskirat409/</li>
</ul>


