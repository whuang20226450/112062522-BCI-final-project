# TODO
demo video / survey

# Data description
Describe the experimental design/paradigm, procedure for 
collecting data, hardware and software used, data size, number of 
channels, sampling rate, the website from which your data was collected, 
owner, source, etc.

In this project, we utilize the dataset presented by
a previous study [1], where an RSVP (Rapid Serial Visual Presentation) paradigm was
used to show 16,740 images to subjects, recording their EEG signals via a 64-channel
EASYCAP system at 1,000 Hz. The dataset consists of around 80,000 samples in total
from each of the 10 subjects, with the task being to classify the EEG signals of the
subjects to one of 27 high-level classes regarding which images he were viewing. For this
project, data from the first subject will be used, and the classes are merged to 14
simplify the task.

**Data preparation**
1. Download the `Raw EEG data` of sub1 from [here](https://osf.io/3jk45/) and Unzip it.
2. Download the `image_metadata.npy` from [here](https://osf.io/qkgtf)
3. Download the `category_mat_manual.mat` from [here](https://osf.io/jum2f/?view_only=)
4. put them under data/LargeEEG/raw and execute `python preprocess.py`. Now we got `0000.set` and `1000.set`. The data files are named in XXXX.set. 
    * First digit: 0 means using all channels without channel selection. 1 means otherwise.
    * Second digit: 0 means not using bandPass filtering. 1 means otherwise.
    * Third digit:
      * 0: no artifact removal method used.
      * 1: ICA
      * 2: ASR
      * 3: autoreject
      * 5: ASR+ICA
    * Fourth digit: 0 means not taking the ERP mean. 1 means otherwise.

5. Use EEGLab for preprocessing to obtain 0100.set 1100.set 1110.set 1120.set 1150.set
6. execute `python preprocess2.py` and input the following 4 digits to generate respective training files.
    * 0000
    * 0020
    * 0100
    * 1000
    * 1100
    * 1110
    * 1120
    * 1130
    * 1150

**Quality evaluation**
1. survey
2. ICA
For `0000.set`

|  Pre-processing       |                          |                          |                    |          | Demographic Attribute   |          |          |            |                 |             |
|:---------------------:|:------------------------:|:------------------------:|:------------------:|:--------:|:-----------------------:|:--------:|:--------:|:----------:|:---------------:|:-----------:|
|  EEG (63 Channels)    | Channel selection        | Bandpass filter (1-50Hz) | ASR                | Brain    | Muscle                  | Eye      | Heart    | Line Noise | Channel Noise   | Other       |
| v                     | x                        | x                        | x                  |          |                         |          |          |            |                 |             |
| v                     | v                        | x                        | x                  |          |                         |          |          |            |                 |             |
| v                     | v                        | v                        | x                  |          |                         |          |          |            |                 |             |
| v                     | v                        | v                        | v                  |          |                         |          |          |            |                 |             |

## Row1 ICLabel
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/55d9f7e8-787c-4990-b539-fda95e0f466a" width="1000">

## Row3 ICLabel
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/e788866f-72f4-48c5-be98-27e2c0258124" height="300">

## Row4 ICLabel
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/ab2fcd22-56fc-44f7-b3ae-bc64059baed9" height="300">

# Introduction
Introduction (2 marks) : Provide an overview of your BCI system, 
dataset, explaining its purpose, functionality, and key features.

# Model Framework
Model Framework (5 marks) : Outline the architecture and 
components of your BCI system. This includes the input/output 
mechanisms, signal preprocessing techniques, data segmentation 
methods, artifact removal strategies, feature extraction approaches, 
machine learning models utilized, and any other relevant components.
![Snipaste_2024-05-10_11-52-41](https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/bff14a68-fd8c-4458-acab-97d12cb0eb97)

# Validation
Validation (3 marks) : Describe the methods used to validate the 
effectiveness and reliability of your BCI system.

# Usage (2 marks) : Describe the usage of their BCI model's code. 
Explain the required environment and dependencies needed to run 
the code. Describe any configurable options or parameters within the 
code. Provide instructions on how to execute the code.

# Results (9 marks): Present a detailed comparison and analysis of 
your BCI system's performance against the competing methods. 
Include metrics such as accuracy, precision, recall, F1-score, or any 
other relevant evaluation metrics. Compare and contrast your BCI 
system with existing competing methods. Highlight the advantages 
and unique aspects of your system

# Reference
