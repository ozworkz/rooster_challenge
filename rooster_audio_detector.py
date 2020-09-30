from utils.rooster_utils import prediction, get_model, load_configurations, set_seed

import torch
import sys
TARGET_SR = 44100

settings = load_configurations(mode="detector")
if(settings == -1):
    print("Error: Failed while loading configurations")
    sys.exit()

set_seed(settings["globals"]["seed"])
melspectrogram_parameters = settings["dataset"]["params"]["melspectrogram_parameters"]

device = torch.device(settings["globals"]["device"])
model = get_model(settings["model"])
model = model.to(device)
model.train(False)

prediction = prediction(test_audio_path="test_audio/rooster_competition.wav",
                        model_config=model,
                        mel_params=melspectrogram_parameters,
                        target_sr=TARGET_SR,
                        threshold=0.4, batch_size=120, period = 0.5, steps=4) # period)




print("Total number of roosters", len(prediction))

standings = prediction.sort_values(by='crow_length_msec', ascending=False)
#print(standings)

print("Duration of crow from each rooster in milliseconds")
for index, rooster in prediction.iterrows():
    print(rooster["rooster_id"], ":", rooster["crow_length_msec"] )

print('\n')

rank = 1
print("Ranking of roosters by crow length")
for index, rooster in standings.iterrows():
    print(rank, ":", int(rooster["rooster_id"]))
    rank += 1

#print("All prediction data")
#print(prediction)
#print(standings)
