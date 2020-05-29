# DeephageTP

DeephageTP: an alignment-free deep learning framework for identifying phage-specific proteins from metagenomic datasets
Version:1.0

# Description

DeephageTP predicts phage-specific proteins using deep learning methods. The method has good prediction precision for phage-specific proteins (TerL,Portal and TerS), and it also can be used to predict sequences from the viromic data. CNN can automatically learn protein sequence patterns and simultaneously build a predictive model based on the learned amino acid sequence patterns.Compared to the alignment methods, the CNN is a natural generalization of amino acid based model, can find the similarities of viral proteins at high latitudes. The more flexible CNN model indeed outperforms the alignment methods on viral protein sequence prediction problem.

# Dependencies

DeephageTP requires Python 3.6 with the packages of numpy, theano, keras and scikit-learn. We recommand the use Conda to install all dependencies. After installing conda,simply run:

conda install python=3.6 numpy theano keras scikit-learn

or creat a virtual environment 

conda create --name deephageTP python=3.6 numpy theano keras scikit-learn 
source activate deephageTP

# Installation

Download the package by

git clone https://github.com/chuym726/DeephageTP.git
cd DeephageTP
