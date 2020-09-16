<img src="https://github.com/jvhuang1786/teslaElonStockpred/blob/master/images/logo.png"></img>

# Using Elon's Tweets to Predict Tesla Stock Price 

## Introduction

This project looked to use Elon's tweets to better predict Tesla stock price using Time Series.  

Financial price data was collected using the Yahoo Finance API

Tweet data was collected using the GetOldTweets3 API 

A webapp was also made for this project and can be found here:

[Tesla Elon Web APP](https://elon-tesla.herokuapp.com)

**Goal of the model was to see if we can use BERT to improve multivariate time series predictions**

### Libraries 

<table>

<tr>
  <td>Hypothesis</td>
  <td>scipy, statsmodel, plotly, numpy, pandas, matplotlib</td>
</tr>

<tr>
  <td>Data Wrangling</td>
  <td>numpy, os, cv2, glob, pandas, PIL, scipy, imageio, keras, augmentor, fastai</td>
</tr>

<tr>
  <td>Data Cleaning</td>
  <td>keras.applications.vgg16, keras.preprocessing.image, matplotlib, numpy</td>
</tr>

<tr>
  <td>Data Visualization</td>
  <td>numpy, os, glob, imageio, time, PIL, keras, matplotlib</td>
</tr>

<tr>
  <td>Classicial Classification of Text</td>
  <td>numpy, scipy, keras, skimage.transform</td>
</tr>

<tr>
  <td>BERT and DistilBert</td>
  <td>numpy, tensorflow, dnnlib, config, train, training, copy, metrics</td>
</tr>

<tr>
  <td>Arima</td>
  <td>numpy, tensorflow, dnnlib, config, train, training, copy, metrics</td>
</tr>

<tr>
  <td>Xgboost and RandomForest regression</td>
  <td>numpy, tensorflow, dnnlib, config, train, training, copy, metrics</td>
</tr>

<tr>
  <td>RNN, LSTM and GRU Neural Networks</td>
  <td>glob, numpy, moviepy, os, PIL, dnnlib, pickle</td>

</table>


## Hypothesis 

Hypothesis using Kruskal- Wallis H-Test and Mann-Whitnney U Test:

     $H_{0}$: the distributions of both samples are equal.

     $H_{A}$: the distributions of both samples are not equal.


* [Hypothesis Notebook](https://github.com/jvhuang1786/teslaElonStockpred/blob/master/notebooks/elonHypothesis.ipynb)



## Data Wrangling

Main steps of in data Augmentation:

     Splitting the data in half
     Combining single image data to split data
     Mirroring the image data    
     Adjusting the brightness of the image data     
     Photoshop increase image resolution
     Resize the image to 512 x 512 and 256 x 256
     Run through fastai
     Use dataset_tool.py from Nvidia to transform to tfrecords


* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)



## Data Cleaning

Main steps of in data Augmentation:

     Splitting the data in half
     Combining single image data to split data
     Mirroring the image data    
     Adjusting the brightness of the image data     
     Photoshop increase image resolution
     Resize the image to 512 x 512 and 256 x 256
     Run through fastai
     Use dataset_tool.py from Nvidia to transform to tfrecords


* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)


## Data Visualization

Main steps of in data Augmentation:

     Splitting the data in half
     Combining single image data to split data
     Mirroring the image data    
     Adjusting the brightness of the image data     
     Photoshop increase image resolution
     Resize the image to 512 x 512 and 256 x 256
     Run through fastai
     Use dataset_tool.py from Nvidia to transform to tfrecords


* [Image Data Cleaning and Augmentation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_dataprep.ipynb)


## Classical Text Classification 

Visualization of Feature Map/Activation Map of images using VGG16 convolutional neural network

     Further from input the less details we can see.

* [Feature Map](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/feature_map.ipynb)

## BERT and DistilBERT

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/mhxx.gif" width="480"></img>

Quick Visualization of image data using a DCGAN requires minimal computing power.

       Generator uses Conv2DTranspose
       Discriminator uses Conv2D
       Hyperparameters:
          Filter
          kernel_size
          Stride
          Padding
          kernel_initializer

* [Deep Convolutional GAN](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/dcgan_mhxx.ipynb)

## XgBoost and Random Forest Regression 

Best way to measure a GAN still is to look at fake images generated however there are quantitative measures.  Such as Frechet Inception Distance, Inception Score and Perceptual Path Length.

       FID measures the normal distribution distance between the real and fake images.  
       The closer the distance, the lower the score the better the image quality and diversity

* [Frechet Inception Distance](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/frechet_inception_distance.ipynb)

## RNN, LSTM, GRU 

You will need to download NVIDIA StyleGAN.  

* [Nvidia StyleGAN](https://github.com/NVlabs/stylegan)

Anime-face dataset was trained for 7 days using the default learning rate and mini batch repeat.  
Need to set the data path for the tfrecords to the location of your tfrecords.  

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/face_morph.gif" width="480"></img>

For MHXX dataset transfer learning was used and learning rate was adjusted to 0.001 and mini batch repeat to 1.
All adjustments were made inside the training/training_loop.py file.  You need to reference in the pickled model as well in resume_run_id and resume_kimg.    

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/armor_morph.gif" width="480"></img>

* [MHXX- StyleGAN](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_stylegan.ipynb)

## Author

* Justin Huang

## Resources

* [Machine Learning Mastery LSTM Neural Networks](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)

* [Time Series using Arima](https://github.com/the-deep-learners/deep-learning-illustrated)

* [Using BERT in PyTorch](https://github.com/the-deep-learners/deep-learning-illustrated)

* [What are RNNs](https://github.com/the-deep-learners/deep-learning-illustrated)

* [What is BERT](https://github.com/the-deep-learners/deep-learning-illustrated)



