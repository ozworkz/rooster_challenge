import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from utils.rooster_utils import set_seed, load_configurations, get_model, get_loaders_for_training

ROOT = Path.cwd()
DATASET_DIR = ROOT / 'dataset'
TRAIN_AUDIO_DIR = DATASET_DIR / 'train_audio'
BIRD_CODE = {'rooster':0}
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

eps = sys.float_info.epsilon

settings = load_configurations(mode="train")
if(settings == -1):
    print("Error: Failed while loading configurations")
    sys.exit()

set_seed(settings["globals"]["seed"])
device = torch.device(settings["globals"]["device"])
output_dir = Path(settings["globals"]["output_dir"])

#configure model
model = get_model(settings["model"])
model = model.to(device)


#configure optimizer
optimizer = getattr(torch.optim, settings["optimizer"]["name"])(model.parameters(), **settings["optimizer"]["params"])

#configure scheduler
scheduler = getattr(torch.optim.lr_scheduler, settings["scheduler"]["name"])(optimizer, **settings["scheduler"]["params"])

#configure loss function
loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

#sigmoid function
sigmoid = nn.Sigmoid()

tmp_list = []

for ebird_d in TRAIN_AUDIO_DIR.iterdir():
    if ebird_d.is_file():
        continue
    for wav_f in ebird_d.iterdir():
        tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])

train_all = pd.DataFrame(
    tmp_list, columns=["ebird_code", "filename", "file_path"])

del tmp_list

print(train_all.shape)
train_file_list=train_all[["file_path", "ebird_code"]].values.tolist()
print("Number of training files: {}".format(len(train_file_list)))

skf = StratifiedKFold(**settings["split"]["params"])

train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.iloc[val_index, -1] = fold_id

use_fold = settings["globals"]["use_fold"]
train_file_list = train_all.query("fold != @use_fold")[["file_path", "ebird_code"]].values.tolist()
val_file_list = train_all.query("fold == @use_fold")[["file_path", "ebird_code"]].values.tolist()

print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file_list), len(val_file_list)))

print(settings["loader"])

num_epochs = settings["globals"]["num_epochs"]
period = settings["globals"]["period"]
threshold = 0.5

def update_metrics(filtered_output, target_cpu, true_positives, false_negatives, true_negatives,false_positives ):
    for i in range(len(filtered_output)):
        if (target_cpu[i] == 1):
            if (filtered_output[i] == target_cpu[i]):
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if (filtered_output[i] == target_cpu[i]):
                true_negatives += 1
            else:
                false_positives += 1
    return true_positives, false_negatives, true_negatives,false_positives

def train_loop(num_epochs=num_epochs):
    #train_loader, valid_loader = get_loaders_for_training(settings["dataset"]["params"], settings["loader"], train_file_list,val_file_list, period=period)

    best_loss = 10000000
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for epoch in range(num_epochs):
        # loader gets random cut from files
        if (epoch % 10 == 0):
            train_loader, valid_loader = get_loaders_for_training(settings["dataset"]["params"], settings["loader"],
                                                                  train_file_list, val_file_list, period=period)
            data_loaders = {"Training": train_loader, "Validation": valid_loader}

        print(f'Epoch #{epoch}')
        for phase in ['Training', 'Validation']:
            print('\n')
            print(f"{phase} is starting")
            print('\n')
            loss_sum = 0.0
            true_positive_sum = 0
            false_positive_sum = 0
            true_negative_sum = 0
            false_negative_sum = 0

            count = 0

            if phase == 'Training':
                model.train()
            else:
                model.eval()

            for batch_idx, (data, target) in enumerate(data_loaders[phase]):
                true_positives = 0
                false_positives = 0
                true_negatives = 0
                false_negatives = 0
                # print(data.shape)
                if phase == 'Training':
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    filtered_output = (sigmoid(output) > threshold)
                    filtered_output = filtered_output.data.to('cpu')
                    target_cpu = target.data.to('cpu')
                    true_positives, false_negatives, true_negatives, false_positives = update_metrics(filtered_output, target_cpu, true_positives, false_negatives, true_negatives, false_positives)
                    loss = loss_func(sigmoid(output), target)
                    loss_sum += loss

                else:
                    with torch.no_grad():
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        filtered_output = (sigmoid(output) > threshold)
                        filtered_output = filtered_output.data.to('cpu')
                        target_cpu = target.data.to('cpu')
                        true_positives, false_negatives, true_negatives, false_positives = update_metrics(
                            filtered_output, target_cpu, true_positives, false_negatives, true_negatives,
                            false_positives)
                        loss = loss_func(sigmoid(output), target)
                        loss_sum += loss

                true_positive_sum += true_positives
                false_positive_sum += false_negatives
                true_negative_sum += true_negatives
                false_negative_sum += false_positives

                count += 1
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}/{num_epochs - 1}, Batch: {batch_idx}, Loss: {loss}, Accuracy = {(true_positives + true_negatives) / (true_positives + false_positives  + true_negatives+ false_negatives + eps)}')
                    print(f'Epoch: {epoch}/{num_epochs - 1}, Precision: {true_positives / (true_positives + false_positives + eps)}, Recall: {true_positives / (true_positives + false_negatives + eps)}')
                    print('\n')

                if phase == 'Training':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # scheduler.step()

            avg_epoch_loss = loss_sum / count
            avg_accuracy = (true_positive_sum + true_negative_sum) / ((true_positive_sum + false_positive_sum + true_negative_sum + false_negative_sum + eps))
            avg_precision = true_positive_sum / (true_positive_sum + false_positive_sum + eps)
            avg_recall = true_positive_sum / (true_positive_sum + false_negative_sum + eps)

            print('=======================================================')
            print(f'=== Epoch {epoch}/{num_epochs - 1} End Results for {phase} ===')
            print('=======================================================')
            print(f'Epoch : {epoch}/{num_epochs - 1}, Loss : {avg_epoch_loss}, Accuracy: {avg_accuracy}')
            print(f'Epoch: {epoch}/{num_epochs - 1}, Precision: {avg_precision}, Recall: {avg_recall}')
            print('=======================================================')

            if phase == 'Validation':
                if ( best_accuracy < avg_accuracy):
                    print("A model with better accuracy, saving...")
                    best_accuracy = avg_accuracy
                    torch.save( model.state_dict(), 'weights/best_accuracy.pth')

                if (best_loss > avg_epoch_loss):
                    print("A model with better loss, saving...")
                    best_loss = avg_epoch_loss
                    torch.save( model.state_dict(), 'weights/best_loss.pth')


train_loop(num_epochs=num_epochs)

print("Saving final model")
torch.save( model.state_dict(), 'weights/final.pth')