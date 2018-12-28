# ekg-risk-models
Code for for fitting EKG-based risk predictors; companion to the paper

> A Comparison of Patient History- and EKG-based Cardiac Risk Scores  
> Andrew C. Miller, Sendhil Mullainathan, Ziad Obermeyer  
> Proceedings of the AMIA Summit on Clinical Research Informatics (CRI), 2018

This repository implements two different EKG-based predictive models

- Resnet: extension of [Rajpurkar et al](https://arxiv.org/abs/1707.01836) convolutional residual neural network model to the multiple lead setting. 

- Beatnet: simple multi-layer perceptron model applied to segmented (and registered) beats. 

The script `analysis.py` trains models, makes plots.

The modules `experiment_data` and `simple_baseline` contain
data loading and model running functions.  The module `ekgmodels` contains
the implementation of the Resnet and Beatnet models.

Note: the structure of the source DataFrame we use will be specified, but is included in this repository. 
