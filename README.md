# TODO
demo video / 

# Data description

In this project, we utilize the dataset from a previous study [1], which employed a Rapid Serial Visual Presentation (RSVP) paradigm to present 16,740 images to subjects while recording their EEG signals using a 64-channel EASYCAP system at 1,000 Hz. The dataset comprises approximately 80,000 samples per subject from 10 subjects in total. The task involves classifying the subjects' EEG signals into one of 27 high-level classes based on the images they were viewing. For this project, we will focus on data from the first subject, and the classes will be consolidated into 14 to simplify the task. The source of the dataset is linked in the `Data preparation` section.

**Quality evaluation**

## ICA ##

|  Pre-processing       |                          |                          |                    |          | Demographic Attribute   |          |          |            |                 |             |
|:---------------------:|:------------------------:|:------------------------:|:------------------:|:--------:|:-----------------------:|:--------:|:--------:|:----------:|:---------------:|:-----------:|
|  EEG (63 Channels)    | Channel selection        | Bandpass filter (1-50Hz) | ASR                | Brain    | Muscle                  | Eye      | Heart    | Line Noise | Channel Noise   | Other       |
| v                     | x                        | x                        | x                  | 32       |  3                      |  1       |  0       |     0      |     6           |  21         |
| v                     | v                        | x                        | x                  | 15       |     0                   |   0      |  1       |     0      |     0           |  1          |
| v                     | v                        | v                        | x                  | 16       |     0                   |   0      |  0       |     0      |     0           |  1          |
| v                     | v                        | v                        | v                  | 16       |     0                   |   0      |  0       |     0      |     0           |  1          |

## Row1 ICLabel (Used in 0000, red ones, possbility > 0.5, are removerd )
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/55d9f7e8-787c-4990-b539-fda95e0f466a" width="1000">

## Row2 ICLabel (Not used)
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/328672e5-27d4-487c-9d0b-2f626b8081cf" height="300">

## Row3 ICLabel (Used in 1110, component 8 is removed)
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/e788866f-72f4-48c5-be98-27e2c0258124" height="300">

## Row4 ICLabel (Used in 1150, component 14 is removed)
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/ab2fcd22-56fc-44f7-b3ae-bc64059baed9" height="300">

# Introduction
EEG data presents challenges due to its low signal-to-noise ratio (SNR). Effective preprocessing is essential for models to accurately interpret EEG data, making artifact removal methods critical for EEG-based EEG-based BCI.

This project aims to evaluate the effectiveness of different artifact removal methods. Utilizing an RSVP EEG visual dataset, we tested various methods such as ASR, ICA, and Autoreject. The results are evaluated based on the performance of EEGNet, with the evaluation metric being the macro F1 score.

# Model Framework

## Preprocessing steps
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/bff14a68-fd8c-4458-acab-97d12cb0eb97" height="300">

As illustrated in the image, the preprocessing starts by selecting 17 channels in the occipital and parietal cortex, similar to the method used in study [1]. Then, a bandpass filter is applied to retain signals within the 1-50 Hz range. After filtering, one of the artifact removal methods is chosen, although it is important to note that the ICU-net is not yet implemented and that ICA refers to the extended Infomax ICA. The signal is then epoched from -0.2s to 0.8s relative to the stimulus onset, followed by baseline correction by subtracting the mean from -0.2s to 0s. Finally, the mean ERP is calculated for epochs sharing the same label, which helps to minimize noise. The data is then split into training, validation, and test sets in a 70:15:15 ratio.

## Model
We utilize EEGNet [2], a compact CNN specifically designed for efficient EEG classification. Its primary advantage lies in its versatility, handling various EEG-based tasks such as motor imagery, ERP classification, and abnormal EEG detection. The lightweight architecture of EEGNet makes it ideal for real-time applications and deployment on devices with limited computational resources. Furthermore, its robustness against noise making it an excellent candidate for our investigation into the necessity of data preprocessing for deep learning methods.

# Validation
To assess the effectiveness of the methods introduced in the lecture, we designed two experiments. In the first experiment, we evaluated steps 1, 2, and 4 using an ablation approach. The second experiment tested the best combination from the first experiment alongside one of four artifact removal methods. The results of these experiments are detailed in the `Results` section and the experiments are evaluated on an independent test set after model tuning to ensure reliability.

# Usage 

**Environment**

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

**Run**

To reproduce experiment1, excute `python exp1.py` and then `python test_exp1.py`.

To reproduce experiment2, excute `python exp2.py` and then `python test_exp2.py`.

In the `experiments` folder, each subfolder contains the training results for specific dataset. For a summary of the performance, you can refer to the `results.txt`.

# Results
your BCI system's performance against the competing methods. 
Include metrics such as accuracy, precision, recall, F1-score, or any 
other relevant evaluation metrics. Compare and contrast your BCI 
system with existing competing methods. Highlight the advantages 
and unique aspects of your system

## Experiment 1
| Dataset ID | Test Accuracy | Test Macro F1 | Test Micro F1 |
|------------|---------------|---------------|---------------|
| 0000       | 0.1176        | 0.0860        | 0.1176        |
| 1100       | 0.1224        | 0.0887        | 0.1224        |
| 1001       | **0.3471**    | 0.2775        | **0.3471**    |
| 0101       | 0.2703        | 0.2543        | 0.2703        |
| 1101       | 0.3137        | **0.2782**    | 0.3137        |

## Experiment 2
| Dataset ID | Test Accuracy | Test Macro F1 | Test Micro F1 |
|------------|---------------|---------------|---------------|
| 1111       | 0.3200        | 0.2982        | 0.3200        |
| 1121       | 0.3484        | **0.3257**    | 0.3484        |
| 1131       | **0.4197**    | 0.3102        | **0.4197**    |
| 1151       | 0.2906        | 0.2689        | 0.2906        |

## Plot

| Category                 | Label |
|--------------------------|-------|
| animal                   | 0     |
| human body               | 1     |
| clothing and accessories | 2     |
| food                     | 3     |
| home and furniture       | 4     |
| kitchen                  | 5     |
| electronics              | 6     |
| medical equipment        | 7     |
| office supply            | 8     |
| musical instrument       | 9     |
| vehicle                  | 10    |
| toy                      | 11    |
| plant                    | 12    |
| other                    | 13    |

### Dataset ID 1111
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/f2009dd2-586c-4cd2-8087-cf0e1cc94a98" height="300">

### Dataset ID 1121
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/ac9c79e1-c9ab-40a3-9bed-fae484732639" height="300">

### Dataset ID 1131
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/f6604bd5-5592-499c-a519-39105eb26cef" height="300">

### Dataset ID 1151
<img src="https://github.com/whuang20226450/112062522-BCI-final-project/assets/29110592/29062230-4afb-4d83-870c-a5f1f9136f1f" height="300">

## Observation
Given the imbalanced nature of the dataset, we use macro F1 as our main metric. From Experiment 2, we observe that the combination labeled `1121` (ASR) performs the best. Adding ICA appears to degrade performance, possibly because the dataset was recorded while the subject was sitting, making ICA an overkill that might remove important components. Interestingly, Autoreject achieves the highest performance in terms of accuracy and micro F1, suggesting that combining it with other methods might yield even better results.

Regarding the performance of each label, labels `1`, `2`, `3`, and `12` (`human body`, `clothing and accessories`, `food`, and `plant`) generally perform better, except in the `1151` combination, which shows lower performance for the `plant` label. These categories are commonly seen objects, which likely contributes to their higher performance. Surprisingly, the `animal` label (`0`) consistently shows low performance across all artifact removal methods. This is unexpected, as one would assume that animals, being more distinct to human perception, would yield better performance.

# Reference
[1] A large and rich EEG dataset for modeling human visual object recognition
[2] EEGNet
