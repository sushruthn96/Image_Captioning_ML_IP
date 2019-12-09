# Neural Networks for Image Captioning

ECE 285 - Machine learning for Image processing. Final project

Description
===========
This is the project "Neural Networks for Image Captioning" developed by the team Neural Net Ninjas composed of Asfiya Baig, Balaji Balachandran and Sushruth Nagesh. 

Requirements
============

Install matplotlib as follows: 

    $ pip install --user matplotlib
    
Install numpy as follows: 

    $ pip install --user numpy
    
Install pillow as follows: 

    $ pip install --user pillow
    
Install argparse as follows: 

    $ pip install --user argparse

Install pycocotools as follows:

    $ pip install --user cython 
    $ pip install --user pycocotools
    
Install nltk as follows:

    $ pip install --user nltk
    
    
Code Organization
=================
demo.ipynb -- Run a demo of our code (reproduces Figure 3 of our report)

build_vocab.py -- Builds the vocabulary using the captions in the training data

resize.py -- Pre-processes the data by resizing it

data_loader.py -- return the data loader for the MSCOCO-2015 dataset

model.py -- Returns the network architecture for the Encoder CNN and the decoder

train.py -- training script

trained_model/encoder.ckpt -- trained encoder 

trained_model/decoder.ckpt --trained decoder


Code Usage
===========

Pre-processing
===============

    $ python build_vocab.py
    $ python resize.py

Training
========

    $ python train.py
    
Testing
========

To test the model, run the demo.ipynb notebook

The trained enocder model can be found [here.](https://drive.google.com/open?id=1fKe_CT5P-nJbyuOeD4Cl22YzJV2tK2et)

The trained deocder model can be found [here.](https://drive.google.com/open?id=1yM34FfMRhF8RXHXqIUUqYMCrtsfKMLv1)
