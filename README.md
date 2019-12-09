# Neural Networks for Image Captioning

ECE 285 - Machine learning for Image processing. Final project

Description
===========
This is the project "Neural Networks for Image Captioning" developed by the team Neural Net Ninjas composed of Asfiya Baig, Balaji Balachandran and Sushruth Nagesh. 

Requirements
============
Install pycocotools as follows:

    $ pip install --user cython 
    $ pip install --user pycocotools
    
Install nltk as follows:

    $ pip install --user nltk
    
    
Code Organization
=================
demo.ipynb -- Run a demo of our code (reproduces Figure 3 of our report)
resize.py -- Pre-processes the data by resizing it
data_loader.py -- return the data loader for the MSCOCO-2015 dataset
model.py -- Returns the network architecture for the Encoder CNN and the decoder
