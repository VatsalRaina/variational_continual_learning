# VARIATIONAL CONTIUAL LEARNING

### By Aymeric Roucher, Vatsal Raina, Shawn Shen and Adian Liusie

This repositry is a reimplemntation of the [variational continual learning](https://arxiv.org/pdf/1710.10628.pdf) paper by Nguyen et. Al. (2018) as part of the MLMI4 course at the university of Cambridge. The code is loosely based on that from the [original author](https://github.com/nvcuong/variational-continual-learning) however with now a pytorch based implementation. 

## Theory

#### Introduction
Continual learning is the area of machine learning where a model is given multiple different tasks that it should try to optimize over and perform well at. However the big caveat in this task is that not all task data is available at the same time: instead the tasks are provided in an online fashion (i.e. the tasks are given one after another) and so the model parameters must be updated incrementally after seeing a specific subset of the task data. Although one may at first consider updating the parameters of a standard deep learning architecture task after task using SGD, it has been shown that deep learning mondels generally suffer from catastrophic forgetting and a model trained in this manner often performs very poorly on early tasks. We therefore want an approach where even though a model may only be trained incrementally on individual tasks, it can still retain relevant information and perform reasonably well over all tasks.

#### Baysian Solution
The VCL paper is built on the observation that Bayesian inference is a very naturally framework for Continual learning. In particular if there are model parameters θ, a prior over the model parameters p(θ) and training data D, assuming independence between training examples the posterior distirbution of the parameters given the data can be written as:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\theta|D_{1:T})&space;\propto&space;p(\theta,&space;D_{1:T})&space;=&space;p(\theta)&space;p(D_{1:T}|\theta)&space;=&space;p(\theta)\prod_{i=1}^{T}p(D_i|\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\theta|D_{1:T})&space;\propto&space;p(\theta,&space;D_{1:T})&space;=&space;p(\theta)&space;p(D_{1:T}|\theta)&space;=&space;p(\theta)\prod_{i=1}^{T}p(D_i|\theta)" title="p(\theta|D_{1:T}) \propto p(\theta, D_{1:T}) = p(\theta) p(D_{1:T}|\theta) = p(\theta)\prod_{i=1}^{T}p(D_i|\theta)" /></a>

Which can be broken further down into:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\theta|D_{1:T-1})&space;p(D_T|\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\theta|D_{1:T-1})&space;p(D_T|\theta)" title="p(\theta|D_{1:T-1}) p(D_T|\theta)" /></a>

Which implies that the distirbution of the model paremeters can be updated after each task, and still lead to exact inference of the true posterior.
## Dependencies

### To install
pip install torch<br />
pip install blitz-bayesian-pytorch
