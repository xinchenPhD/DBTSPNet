## DBTSP-Net

## Paper Title
DBTSP-Net: A Temporal-Spatial Parallel Network with Optuna Optimization for Subject-Specfic Motor Imagery EEG Decoding and Visualiazation

## Keywords
Motor imagery (MI), Brain-computer interface (BCI), Electroencephalogram (EEG) decoding, Temporal-Spatial parallel network, Transformer, Adaptive weighted feature fusion

## Requirements
- Python 3.7.0
- Pytorch 1.13.1
- MATLAB R2013b

## Datasets
- The experiments were carried out on the public datasets 2a and 2b of BCI Competition IV , provided by Graz University of Technology. These public datasets are publicly available  and can be accessed via the following links: www.bbci.de/competition/iv/
- [BCI_competition_IV2a](https://www.bbci.de/competition/iv/desc_2a.pdf) 
- [BCI_competition_IV2b](https://www.bbci.de/competition/iv/desc_2b.pdf) 

## Data preprocessing
The preprocessing steps for the datasets are as follows: (see BCIIV2a.m and BCIIV2b.m in floder "preprocessing")
- Convert from .GDF format to .Mat format using open-source toolbox Biosig（http://biosig.sourceforge.net/）
- Filtering
-  Perform Z-score standardization for normalization
The standardized datas are shown in folder "standardization_BCIIV_2a_data" and “standardization_BCIIV_2b_data”.

## MI-EEG models
Our proposed algorithm and benchmark models are shown in folder "MI-EEG models”.
- Our proposed algorithm（DBTSP-Net）
- ShallowConvNet
- DeepConvNet
- EEGNet
- Conformer
- CNN-LSTM
- CNN-BiLSTM
- CTNet
- CNN-Gated Transformer
- CNN-Mamba(Linux recommended）

## Performance Evaluation
We evaluate the models based on the following metrics:
- accuracy
- kappa 
- Leave-One-Subject-Out (LOSO) Cross Validation

## Visualiaztion
The Visualiaztion code implementation is shown in folder “visualization”.
- t-SNE
- EEG topological map
