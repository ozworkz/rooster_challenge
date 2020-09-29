# Rooster Challenge

This repo contains a solution for detecting rooster sounds from audio files.

## Instructions

### Install requirements:
  - **pip3 install -r requirements.txt**
  
### Rooster Detection 
  - Run **rooster_audio_detector.py** file. You can change the weight settings in **detector_conf.yaml** if you want to run the code with your own training.
  
### Training 
  
1. Run **download_prepare_dataset.py** file to download ESC-50: Environmental Sound Classification dataset. The script will generate an augmented rooster and background sounds to be trained on.

More information on dataset can be found from https://github.com/karolpiczak/ESC-50

2. Run **train.py** file to start training. Optionally you can adjust settings in **train_conf.yaml** file.
