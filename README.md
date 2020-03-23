# 3db - Explore database structure in 3D environments.  
> A Python implementation of CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy.
It was developed by Marzio Monticelli as final project for the course of Neural Network at Sapienza, University of Rome.

This project is about an implementation of the paper by Microsoft Research group
in collaboration with Princeton University called "CryptoNets: Applying Neural
Network to Encrypted Data with High Throughput and Accuracy" by Nathan Dowlin,
Ran Gilad-Bachrach, Kim Laine, Kristin Lauter, Micheal Naehrig and John Wernsing
published on February 24, 2016. The paper is about applying machine learning
to a problem involves medical, financial, or other types of sensitive data requiring
accurate predictions and careful attention to maintaining data privacy and security.
In their work a method to convert learned neural networks to CryptoNets is presented.
This project is about an implementation written in Python of the functionalities
presented in CryptoNets paper. Results presented in this work are elaborated on the
MNIST optical character recognition tasks and are compared to the results from the
original paper so that to demonstrate the quality of both throughput and accuracy
with respect to the results from the original work by Microsoft Research group. . .


## What is this repository for?
This repository is intended for presentation purposes only.

## Environment

The provided code was built and tested on a Linux Mint 19.2 (Tina)- Ubuntu bionic machine with 64-pc-linux-gnu (x86) architecture.
In particular the following libraries (or compatible) are required:

* gcc (Ubuntu 7.4.0-1ubuntu1 18.04.1) >= 7.4.0
* g++ (Ubuntu 7.4.0-1ubuntu1 18.04.1) >= 7.4.0
* clang >= 6.0.0-ubuntu2
* anaconda (64-bit x86) >= 3.7
* keras-applications >= 1.0.8 » keras
* matplotlib >= 3.1.1
* numpy >= 1.17.3
* pillow >= 6.2.1
* pyfhel >= 2.0.1
* Python >= 3.7.5
* scikit-learn >= 0.21.3
* tensorflow-gpu >= 2.0.0 » tensorflow-gpu
* tensorboard >= 2.0.0 » tensorflow-gpu


Anancoda3 is used to manage libraries, dependencies and Python environments.
Through Anaconda Distribution you can easily create a custom environment containing
all the main libraries used in this work.

## Installation

Once you are sure your system match the requirements contained in Environment
section you are ready to install Anaconda.
You can download Anaconda from the official site (https://www.anaconda.com) :
be sure to download Anaconda3 (version 3.7 or grater) compatible with your system.
If you need a complete guide to install and use Anaconda on your local machine,
you can refer to the Anaconda official documentation (https://docs.anaconda.com);
inside the official documentation you will find all the information you need to complete
this task.

Anaconda makes it easy to install TensorFlow, enabling your data science, machine
learning, and artificial intelligence workflows.
This page (https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow) shows
how to install TensorFlow with the conda package manager included in Anaconda
and Miniconda. TensorFlow with conda is supported on 64-bit Windows 7 or later,
64-bit Ubuntu Linux 14.04 or later, 64-bit CentOS Linux 6 or later, and macOS 10.10
or later. The instructions are the same for all operating systems. No apt install or
yum install commands are required.

First of all you need to create a new environment containing the current release
of TensorFlow GPU (if supported). You can do it opening the Anaconda command
prompt or PowerShell and typing the following two commands:

```sh
conda create -n cryptonets tensorflow-gpu
conda activate cryptonets
```

Your environment with TensorFlow GPU is now installed and ready to use.
If your machine does not support the TensorFlow GPU version, you can install the
default version of the library since GPU functionalities are not strictly required.

Once your environment contains TensorFlow (GPU) and is it active in the console
you can easily install all the libraries are required by this project you can find in the
previous section. You can install all but pyfhel library with the conda command as follow
(be sure your cryptonets environment is active in your console) :

```sh
conda install keras
conda install matplotlib
...
conda install scikit-learn
```

If you need to force the installation of a specific library (matplotlib in this example)
you can run the following command:

```sh
conda install matplotlib --force
```

Finally you need to install the latest release of Pyfhel (vrs 2.0.1 or grater) through
pip package manager since it is not available in the Anaconda Cloud.

```sh
pip install Pyfhel
```

You can see the documentation related to Pyfhel in the official Github page (https://github.com/ibarrond/Pyfhel).

To complete the installation process you have to extract the provided folder in the
directory created for your environment.

## Release History

* 1.0
    * The first proper release
    * The project is for presentation prupose only.
    * There will be no further releases at the moment

## Meta

Marzio Monticelli – marzio.monticelli@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/MarzioMonticelli/python-cryptonet](https://github.com/MarzioMonticelli/python-cryptonet)
