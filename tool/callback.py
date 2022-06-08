import json
import os
import torch

LOG_PATH = 'log/result.json'
MODEL_PATH = 'model/saved_model/model.pt'


def save_log(data):
    metrics = data['metrics']
    model = data['model']
    to_json(metrics)
    if data['epoch'] % 10 == 0:
        torch.save(model.state_dict(), MODEL_PATH)


def to_json(metrics):
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            json_ = json.load(f)
    else:
        json_ = []
    json_.append(metrics)

    with open(LOG_PATH, 'w') as f:
        json.dump(json_, f, indent=4, ensure_ascii=False)
