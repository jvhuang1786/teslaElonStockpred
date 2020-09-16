<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/fakes007560.png"></img>

# Generative Adversarial Networks Character Concept Creation

## Second Capstone Project For Springboard using neural networks to generate images

## Introduction

This project goes through 3 types of GAN. Vanilla GAN, Deep Convolutional GAN and Style GAN.

Anime face data was collected through Kaggle. It contained 85000 images

https://www.kaggle.com/splcher/animefacedataset

MHXX armor data was collected through a fans website.  It contained 1083 images. After Augmentation became 19437.

http://mhxx-soubigazou.info

**Goal is to create a model that can generate images based on old assets by the studio or company and thus create future assets faster and at a lower cost**

### Computer Requirements/Libraries/Modules

Need to install the following

<table>

<tr>
  <td>Computer Requirements</td>
  <td>Nvidia GPU 11 gb and above, tensorflow-gpu 1.1.5, cuda 10.0, cuDNN 7.5, tensorrt 5.6.5</td>
</tr>

<tr>
  <td>Image Data Cleaning and Augmentation</td>
  <td>numpy, os, cv2, glob, pandas, PIL, scipy, imageio, keras, augmentor, fastai</td>
</tr>

<tr>
  <td>Feature Map Exploration</td>
  <td>keras.applications.vgg16, keras.preprocessing.image, matplotlib, numpy</td>
</tr>

<tr>
  <td>Deep Convolutional Generative Adversarial Networks</td>
  <td>numpy, os, glob, imageio, time, PIL, keras, matplotlib</td>
</tr>

<tr>
  <td>Frechet Inception Distance</td>
  <td>numpy, scipy, keras, skimage.transform</td>
</tr>

<tr>
  <td>Style GAN (You will have to go to NVIDIA to download Style GAN and place this file inside it)</td>
  <td>numpy, tensorflow, dnnlib, config, train, training, copy, metrics</td>
</tr>

<tr>
  <td>Image generator and movie clip</td>
  <td>glob, numpy, moviepy, os, PIL, dnnlib, pickle</td>

</table>


## Image Data Cleaning and Augmentation

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


## Feature Map Exploration

Visualization of Feature Map/Activation Map of images using VGG16 convolutional neural network

     Further from input the less details we can see.

* [Feature Map](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/feature_map.ipynb)

## Deep Convolutional Generative Adversarial Network

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

## Frechet Inception Distance

Best way to measure a GAN still is to look at fake images generated however there are quantitative measures.  Such as Frechet Inception Distance, Inception Score and Perceptual Path Length.

       FID measures the normal distribution distance between the real and fake images.  
       The closer the distance, the lower the score the better the image quality and diversity

* [Frechet Inception Distance](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/frechet_inception_distance.ipynb)

## StyleGAN

You will need to download NVIDIA StyleGAN.  

* [Nvidia StyleGAN](https://github.com/NVlabs/stylegan)

Anime-face dataset was trained for 7 days using the default learning rate and mini batch repeat.  
Need to set the data path for the tfrecords to the location of your tfrecords.  

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/face_morph.gif" width="480"></img>

For MHXX dataset transfer learning was used and learning rate was adjusted to 0.001 and mini batch repeat to 1.
All adjustments were made inside the training/training_loop.py file.  You need to reference in the pickled model as well in resume_run_id and resume_kimg.    

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/armor_morph.gif" width="480"></img>

* [MHXX- StyleGAN](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/mhxx_stylegan.ipynb)

## Image Generation and Movie Clip

To generate images and put them into an mp4 file.  
    
    Generates 1000 random images from pickled file
    Can stitch together images and put them into an mp4 to see. 

* [Image Generation](https://nbviewer.jupyter.org/github/jvhuang1786/mhxxCapStone/blob/master/img_generate.ipynb)

<img src="https://github.com/jvhuang1786/mhxxCapStone/blob/master/images/vidstitch_image_071215_20200315.jpg" width="480"></img>


## Author

* Justin Huang

## Resources

* [Machine Learning Mastery](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)

* [Deep Learning Illustrated](https://github.com/the-deep-learners/deep-learning-illustrated)

* [DCGAN](https://towardsdatascience.com/generate-anime-style-face-using-dcgan-and-explore-its-latent-feature-representation-ae0e905f3974)

* [Nvidia Style GAN](https://arxiv.org/abs/1812.04948)

* [Anime Faces with Gwern](https://www.gwern.net/Faces)
