import torchsummary

from model.odas import ODAS
from model.loss import Loss
from tool.dataset import data_deal
from torch_lib import traverse
from tool.metrics import accuracy, precision, recall, HM, DSC, IOU, f1
import torch
import os
from PIL import Image
import numpy as np
import json


model = ODAS()
model.load_state_dict(torch.load('model/saved_model/model.pt'))
dataset, neg_rate, _ = data_deal(batch_size=1, val_ratio=0, train_ratio=0, shuffle=False, random=False)
loss = Loss(weight=torch.tensor([1 - neg_rate, neg_rate]))
img_path = os.listdir('dataset/imgs')


def callback(data):
    y_pred = data['y_pred']  # 模型预测
#     y_pred = (y_pred[0] + y_pred[1] + y_pred[2] + y_pred[3]) / 4  # 这里需要改一下，模型的输出不一样
    print(y_pred.shape)
    print(data['metrics'])  # 计算的metrics结果
    print(img_path[data['step']])  # 通过当前step索引得到文件路径
    path = img_path[data['step']]
    img = np.uint8(torch.argmax(y_pred[0], dim=0) * 255)
    Image.fromarray(img).save('dataset/pred_result/' + path)

    file = 'dataset/pred_result/' + path[0:path.find('.')] + '.json'
    with open(file, 'w') as f:
        json.dump(data['metrics'], f, indent=4, ensure_ascii=False)


traverse(model, dataset, callbacks=[callback], metrics=[(loss, 'loss'), accuracy, precision, recall, HM, DSC, IOU, f1], console_print=True)
