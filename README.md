# VARIATIONAL CONTIUAL LEARNING

### By Aymeric Roucher, Vatsal Raina, Shawn Shen and Adian Liusie

This repositry is a reimplemntation of the [variational continual learning](https://arxiv.org/pdf/1710.10628.pdf) paper by Nguyen et. Al. (2018) as part of the MLMI4 course at the university of Cambridge. The code is loosely based on that from the [original author](https://github.com/nvcuong/variational-continual-learning) however with now a pytorch based implementation. 

## Theory

#### Introduction
Continual learning is a type of learning where a model is given multiple different tasks that it should try to optimizer however. However the tasks are not all available at the start of training, and instead the tasks are provided in an online fashion (i.e. the tasks are given one after another). Althouhg one may at first consider updating the parameters of a standard deep learning architecture task after task using SGD, general deep learning mondels are shown to suffer from catastrophic forgetting and the model often performs very poorly on early tasks. We therefore want an approach where even though we may train the model incrementally on different tasks, it should still retain relevant information from earlier tasks.

#### Baysian Solution
A good insight into 


## Dependencies

### To install
pip install torch<br />
pip install blitz-bayesian-pytorch
