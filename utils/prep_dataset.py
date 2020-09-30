import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.model_selection import StratifiedKFold
import typing as tp

def generate_rooster_dataset(augment_option: tp.Dict):
    ecs_csv_path = os.path.abspath(os.path.expanduser('dataset/ESC-50-master/meta/esc50.csv'))
    ECS_DATA_DIR = os.path.abspath(os.path.expanduser('dataset/ESC-50-master/audio'))
    DATASET_DIR = os.path.abspath(os.path.expanduser('dataset/train_audio'))
    ROOSTER_DATA_DIR = os.path.join(DATASET_DIR, 'rooster')
    BACKGROUND_DIR = os.path.join(DATASET_DIR, 'background')

    seed = 42
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    ECS50data = pd.read_csv(ecs_csv_path)
    roosterData = ECS50data[ECS50data['category'] == 'rooster']
    notRoosterData = ECS50data[ECS50data['category'] != 'rooster']

    #skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    #for fold_id, (train_index, val_index) in enumerate(skf.split(notRoosterData, notRoosterData["category"])):
    #    notRoosterData.iloc[val_index, 1] = fold_id

    use_fold = 0
    background_file_list = list(notRoosterData["filename"]) #.query("fold == @use_fold")["filename"].values.tolist()
    rooster_file_list = list(roosterData["filename"])

    y_roosters = []
    sr_roosters = []

    y_background = []
    sr_background = []

    for filename in rooster_file_list:
        # y, sr = librosa.load(os.path.join(ROOSTER,filename))
        y, sr = sf.read(os.path.join(ECS_DATA_DIR, filename))
        y = librosa.effects.trim(y)
        y_roosters.append(y)
        sr_roosters.append(sr)

    for filename in background_file_list:
        # y, sr = librosa.load(os.path.join(BACKGROUND,filename))
        y, sr = sf.read(os.path.join(ECS_DATA_DIR, filename))
        y_background.append(y)
        sr_background.append(sr)

    print("Total rooster files found: ", len(y_roosters))
    print("Total background files to be used: ", len(sr_background))

    print(augment_option)
    if (augment_option["set"]["all"] == 1):
        print("Creating augmented rooster and background sound datasets...")
        if not os.path.exists(ROOSTER_DATA_DIR):
            os.makedirs(ROOSTER_DATA_DIR)

        if not os.path.exists(BACKGROUND_DIR):
            os.mkdir(BACKGROUND_DIR)

        for i in range(int(len(y_background) * augment_option["n_sample_coef"])):
            r1 = np.random.randint(len(y_roosters))
            y1 = np.random.randint(len(y_background))
            y2 = np.random.randint(len(y_background))
            # y3 = np.random.randint(len(y_background))
            # y4 = np.random.randint(len(y_background))

            aug_sample_wi_rooster = 2 * y_roosters[r1][0] + y_background[y1][0:(len(y_roosters[r1][0]))] + y_background[
                                                                                                               y2][0:(
                len(y_roosters[r1][0]))]  # + y_background[y3][0:(len(y_roosters[r1][0]))]
            aug_sample_no_rooster = y_background[y1][0:(len(y_roosters[r1][0]))] + y_background[y2][0:(
                len(y_roosters[r1][0]))]  # + y_background[y3][0:(len(y_roosters[r1][0]))]

            librosa.output.write_wav(f'{ROOSTER_DATA_DIR}/{i}.wav', aug_sample_wi_rooster, 44100)
            librosa.output.write_wav(f'{BACKGROUND_DIR}/{i}.wav', aug_sample_no_rooster, 44100)

    else:
        if (augment_option["set"]["rooster"] == True):
            for i in range(int(len(y_background) * augment_option["n_sample_coef"])):
                r1 = np.random.randint(len(y_roosters))
                y1 = np.random.randint(len(y_background))
                y2 = np.random.randint(len(y_background))
                # y3 = np.random.randint(len(y_background))
                # y4 = np.random.randint(len(y_background))

                aug_sample_wi_rooster = 2 * y_roosters[r1][0] + y_background[y1][0:(len(y_roosters[r1][0]))] + \
                                        y_background[y2][
                                        0:(len(y_roosters[r1][0]))]  # + y_background[y3][0:(len(y_roosters[r1][0]))]

                librosa.output.write_wav(f'{ROOSTER_DATA_DIR}/{i}.wav', aug_sample_wi_rooster, 44100)

        if (augment_option["set"]["background"] == True):
            for i in range(int(len(y_background) * augment_option["n_sample_coef"])):
                r1 = np.random.randint(len(y_roosters))
                y1 = np.random.randint(len(y_background))
                y2 = np.random.randint(len(y_background))
                # y3 = np.random.randint(len(y_background))
                # y4 = np.random.randint(len(y_background))

                aug_sample_no_rooster = y_background[y1][0:(len(y_roosters[r1][0]))] + y_background[y2][0:(
                    len(y_roosters[r1][0]))]  # + y_background[y3][0:(len(y_roosters[r1][0]))]

                librosa.output.write_wav(f'{BACKGROUND_DIR}/{i}.wav', aug_sample_no_rooster, 44100)
