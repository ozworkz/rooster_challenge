# Rooster Challenge

This repo contains a solution for detecting rooster sounds from audio files.

## Instructions

### Install requirements:
  - **pip3 install -r requirements.txt**
  
### Rooster Detection 
  - Run **rooster_audio_detector.py** file. You can change the weight settings in **detector_conf.yaml** if you want to run the code with your own training weights.
  
### Training 
  
1. Run **download_prepare_dataset.py** file to download ESC-50: Environmental Sound Classification dataset. The script will generate an augmented rooster and background sounds to be trained on. More information on dataset can be found from https://github.com/karolpiczak/ESC-50

  - For transfer learning purposes, I used resnest model which was also provided good results in Kaggle's bird call classification competition https://www.kaggle.com/c/birdsong-recognition . The weights I used in the training can found in the repo. 

 - There are two modes of training; **train_with_birdcall** and **continue2train** . If train_with_birdcall is configured on **train_conf.yaml** file, then set weights option in train_conf.yaml as **resnet_bird_call**. If continue2train is chosen for training mode, you can set any of the output weights; **best_loss**, **best_accuracy** or **final**.
 

2. Run **train.py** file to start training. Optionally you can adjust training settings(number of epochs, batch_size, ...) in **train_conf.yaml** file.  
