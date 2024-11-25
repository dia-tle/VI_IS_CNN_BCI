# VI_IS_CNN_BCI
Summary: Adapted 1D-CNN model (Python) for classification of EEG data for Visual Imagery and Imagined Speech mental paradigms. Includes MNE Python pre-processing script, R script for data analysis.


Introduction: 

This is a comprehensive script package for my research project "Classification of Visual Imagery and Imagined Speech EEG based Brain Computer Interfaces using 1D Convolutional Neural Network" as part of my submission for a MSc in Computational Cognitive Neuroscience. The 1D-CNN model was adapted from Mattioli et al (https://doi.org/10.1088/1741-2552/ac4430.)'s for Motor Imagery EEG classification to Visual Imagery and Imagined Speech EEG data classification. Notable changes include feature extraction of frequency domain correlating to the mental paradigms of interest, VI (alpha and beta) and IS (beta and gamma). The data collected using 64-channel BioSemi ActiveTwo cannot be open source, but I'm making the scripts available for those who want to explore classification of EEG VI and IS using a CNN.

This study took a comprehensive approach to build on the current foundation of knowledge of VI combined with IS to optimize the decoding performance for BCI use and classification by utilising an adapted 1D-CNN. Moreover, to provide insights into BCI illiteracy and optimal performance in novice users it was explored if there are any correlations of cognitive, behavioural, or psychological factors that can predict VI BCI performance by using the Vividness of Visual Imagery Questionnaire (VVIQ).

Hypotheses:

Hypothesis 1: During the VI task of visualisation of “push” condition, alpha and beta suppression would be observed in comparison to “relax” condition. During the IS task of covert speech of “push” word, gamma oscillations in the temporal and/or dominant speech and language areas of Broca and Wernicke’s.

Hypothesis 2: Those that score higher on VVIQ will show more affinity towards vividness and imaginary thinking and so will perform better on the VI paradigm compared to those who score lower on VVIQ in novice BCI users.

Hypothesis 3: The 1D-CNN performance would be able to be replicated as in Mattioli et al. for the VI and IS data. Based on the parameters used in that study, such as checkpointing, early stopping and data augmentation utilising the adapted 1D-CNN on this dataset would reproduce similar performance. Whether there would be better performance in one paradigm over the other. This would be measured by the cluster-based permutation statistical analysis and through the CNN’s ability to classify accurately the different neural changes between conditions in VI and IS paradigms.

Explanation of the scripts: 

Pre-processing of EEG data: 
- MNEPreprocessing.py

Statistical analyses of VVIQ results: 
- Stats.R

- Time Frequency Analysis - Morlet Wavelet Convolution for feature extraction (not provided)
  
- Nonparametric Cluster-based Permutation Analysis (not provided) 

1D-CNN model for Classification for EEG data: 
- Original model - model.py 
- CNN model - CNN.py 
- Visual Imagery model - methodvi.py and Vi_train.py
- Imagined Speech model - modelis.py and IS_train.py


You can find my thesis here: 

