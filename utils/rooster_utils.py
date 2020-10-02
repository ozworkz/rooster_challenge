import os
import random
import typing as tp

import yaml
from pathlib import Path
import warnings
import time
from contextlib import contextmanager

import cv2
import librosa
import soundfile as sf

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch
import pandas as pd
import sys


def load_configurations(mode="train"):
    if(mode == "detector"):
        config_file = "detector_conf.yaml"
    elif(mode == "train"):
        config_file = "train_conf.yaml"
    elif(mode == "dataset_generator"):
        config_file = "generate_dataset_conf.yaml"

    else:
        print("Error: Unknown configurations mode")
        return -1

    with open(config_file, 'r') as stream:
        try:
            settings= yaml.safe_load(stream)
            #print(settings)
        except yaml.YAMLError as exc:
            print(exc)

    return settings

@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: igno


# Creates an image representation(V) of matrix to be used in resnest
def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


class SpectrogramDataset(data.Dataset):
    def __init__(
            self,
            file_list: tp.List[tp.List[str]], period, img_size=224,
            waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={},
    ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.period = period

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, ebird_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        # Size adjustments to ensure that all samples have the same length
        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = int(sr * self.period)
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                print("Starting_to_sample from: ", start)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        label = np.zeros(1, dtype="f")
        # labels[BIRD_CODE[ebird_code]] = 1
        if (ebird_code == 'rooster'):
            label[0] = 1
        else:
            label[0] = 0

        return image, label

def get_loaders_for_training(
        args_dataset: tp.Dict, args_loader: tp.Dict,
        train_file_list: tp.List[str],
        val_file_list: tp.List[str],
        period
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, period=period, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, period=period, **args_dataset)

    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    valid_loader = data.DataLoader(val_dataset, **args_loader["val"])

    return train_loader, valid_loader


def get_model(args: tp.Dict):
    model = getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    del model.fc
    weigths_path = os.path.abspath(os.path.expanduser(f'weights/{args["weights"]}.pth'))
    mode = args["mode"]

    if(mode == "train_with_birdcall"):
        #print(f"Model configuration mode: {mode}")
        model.load_state_dict(torch.load(weigths_path), strict=False)

    #for param in model.parameters():
    #    param.requires_grad = False

    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        # nn.Linear(1024,  (args["params"]["n_classes"])))
        nn.Linear(1024, 1))

    if(mode == "detector" or mode =="continue2train"):
        #print(f"Model configuration mode: {mode}")
        if os.path.exists(weigths_path):
            model.load_state_dict(torch.load(weigths_path), strict=True)
        else:
            sys.exit(f"Weight file {weigths_path} can't be found\nYou can download weights from https://drive.google.com/drive/folders/1BrarBYMUH1V4qlMMzU_dT6WQqPhj9J6q")

    return model


class TestDataset(data.Dataset):
    def __init__(self, clip: np.ndarray, img_size=224, melspectrogram_parameters={}, sr=44100, period=0.5, steps=500):
        self.clip = clip
        self.img_size = img_size
        self.melspectrogram_parameters = melspectrogram_parameters
        self.sr = sr
        self.period = period
        self.steps = steps  # change this

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        y = self.clip.astype(np.float32)
        len_y = len(y)
        start = 0
        end = start + int(self.sr * self.period)
        images = []
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (self.sr * self.period):
                break
            start = start + int(self.sr * self.period * 1 / self.steps)
            end = start + int(self.sr * self.period)

            melspec = librosa.feature.melspectrogram(y_batch, sr=self.sr, **self.melspectrogram_parameters)
            melspec = librosa.power_to_db(melspec).astype(np.float32)
            image = mono_to_color(melspec)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)
            images.append(image)

        #print("Number of images", len(images))
        images = np.asarray(images)
        return images


def prediction_for_clip(clip: np.ndarray,
                        model: None,
                        mel_params: dict,
                        threshold=0.95, batch_size=15, period=0.5, steps=500):
    dataset = TestDataset(clip=clip, img_size=224, melspectrogram_parameters=mel_params, period=period, steps=steps)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    prediction_dict = {}

    image_time = 0
    batch_time = 0
    num_rooster = 0

    for image in loader:  # progress_bar(loader):
        # to avoid prediction on large batch
        image = image.squeeze(0)
        whole_size = image.size(0)
        if whole_size % batch_size == 0:
            n_iter = whole_size // batch_size
        else:
            n_iter = whole_size // batch_size + 1

        all_events = set()

        last_period_object_prob = 0.0

        for batch_i in range(n_iter):
            batch = image[batch_i * batch_size:(batch_i + 1) * batch_size]
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)

            batch = batch.to(device)
            with torch.no_grad():
                prediction = F.sigmoid(model(batch))
                proba = prediction.detach().cpu().numpy()



            # Moving average as a low pass filter
            object_prob_filtered = np.zeros_like(proba)


            window_size = 5
            for i in range(proba.shape[0] - window_size + 1):
                for j in range(window_size):
                    object_prob_filtered[i] += 1 / window_size * proba[i + j]

            for i in range(proba.shape[0] - window_size + 1, proba.shape[0]):
                window_size -= 1
                for j in range(window_size):
                    object_prob_filtered[i] += 1 / window_size * proba[i + j]

            mask = object_prob_filtered >= threshold

            filtered = proba.copy()

            #print('{:02d}:{:02d}.{:04d}'.format(int(batch_time/1000/60),int(batch_time/1000)%60,batch_time%1000))
            #print(filtered)
            object_prob_filtered = object_prob_filtered * mask
            #print(object_prob_filtered)

            time_idx = 0
            for i in range(len(object_prob_filtered)):

                # If there is a dramatic change in probability of object, it means there is a new object in this period
                if ((object_prob_filtered[i] - last_period_object_prob) >= threshold):
                    num_rooster += 1
                    prediction_dict.update({num_rooster: {'start': batch_time + time_idx,  # + int((period) * 1000/2),
                                                          'stop': batch_time + time_idx,  # + int((period) * 1000/2),
                                                          'length': 0}})
                elif ((object_prob_filtered[i] - last_period_object_prob) <= - threshold):
                    prediction_dict[num_rooster]['stop'] = batch_time + time_idx  # + int((period) * 1000/2)
                    prediction_dict[num_rooster]['length'] = prediction_dict[num_rooster]['stop'] - \
                                                             prediction_dict[num_rooster][
                                                                 'start']  # - int( 1 * period *1000)

                last_period_object_prob = object_prob_filtered[i]
                time_idx += int((period) * 1000 / steps)  # miliseconds

            batch_time += int(batch_size * (period) * 1000 / steps)  # miliseconds

        # image_time += int(1000*period/steps)
    return prediction_dict


def prediction(test_audio_path: Path,
               model_config: None,
               mel_params: dict,
               target_sr: int,
               threshold=0.95, batch_size=30, period=0.5, steps=500):
    model = model_config
    warnings.filterwarnings("ignore")

    prediction_dfs = []
    #with timer(f"Loading {test_audio_path}"):
    clip, _ = librosa.load(test_audio_path, sr=target_sr, mono=True, res_type="kaiser_fast")

    #with timer(f"Prediction on {test_audio_path}"):
    prediction_dict = prediction_for_clip(clip=clip,
                                              model=model,
                                              mel_params=mel_params,
                                              threshold=threshold, batch_size=batch_size, period=period, steps=steps)

    prediction_df = pd.DataFrame(columns=['rooster_id', 'crow_start', 'crow_stop', 'crow_length_msec'])
    for i in prediction_dict:
        prediction_df = prediction_df.append({'rooster_id': i,
                                              'crow_start': '{:02d}:{:02d}.{:04d}'.format(
                                                  int(prediction_dict[i]['start'] / 1000 / 60),
                                                  int(prediction_dict[i]['start'] / 1000) % 60,
                                                  prediction_dict[i]['start'] % 1000),
                                              'crow_stop': '{:02d}:{:02d}.{:04d}'.format(
                                                  int(prediction_dict[i]['stop'] / 1000 / 60),
                                                  int(prediction_dict[i]['stop'] / 1000) % 60,
                                                  prediction_dict[i]['stop'] % 1000),
                                              'crow_length_msec': int(prediction_dict[i]['length'])},
                                             ignore_index=True)

    prediction_df = prediction_df[prediction_df["crow_length_msec"] > 400]

    return prediction_df

