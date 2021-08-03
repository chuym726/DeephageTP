# # About DeephageTP:

DeephageTP: an alignment-free deep learning framework for identifying phage-specific proteins from metagenomic datasets. DeephageTP first extracts feature from a raw sequence by convolution layer and makes a precision based on the features.

Deephage Version:1.0

# # Description:

DeephageTP predicts phage-specific proteins using deep learning methods. The method has good prediction precision for phage-specific proteins (TerL,Portal and TerS), and it also can be used to predict sequences from the viromic data. CNN can automatically learn protein sequence patterns and simultaneously build a predictive model based on the learned amino acid sequence patterns.Compared to the alignment methods, the CNN is a natural generalization of amino acid based model, can find the similarities of viral proteins at high latitudes. The more flexible CNN model indeed outperforms the alignment methods on viral protein sequence prediction problem.

## workflow:
https://github.com/chuym726/DeephageTP/blob/master/figure%201A.png

![Figure](https://github.com/chuym726/DeephageTP/blob/master/figure%201A.png?raw=true)


# # Installation:

DeephageTP is implemented in with [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) library. For detail instruction of installing [Tensorflow](https://www.tensorflow.org/install) and [Keras](https://keras.io/).

## Dependencies:

* Python: 3.6
* Tensorflow: 1.5.0 
* keras: 2.2.2
* numpy: 1.15.4 
* scikit-learn: 0.20.1

## Usage:
Clone the repository or download compressed source code files. 

```
git clone https://github.com/chuym726/DeephageTP.git

cd DeephageTP
```

## Install dependencies:

```
conda install python=3.6 numpy theano Keras=v2.2.2 scikit-learn Prodigal=v2.6.2

or creat a virtual environment 

conda create --name deephageTP python=3.6 numpy theano Keras=v2.2.2 scikit-learn Prodigal=v2.6.2

source activate deephageTP
```


# Data:
All data used by experiments described in manuscript is available at [Github](https://github.com/chuym726/DeephageTP).

# Citation:
Yunmeng Chu, Shun Guo, Dachao Cui, Xiongfei Fu, Yingfei Ma. DeephageTP: A Convolutional Neural Network Framework 1 for Identifying Phage-specific Proteins from metagenomic sequencing data. 

# Contact:

If you have any question, please contact me by email (ymchu1990@gmail.com) 

