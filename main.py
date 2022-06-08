import torch.cuda
import numpy as np
import torchvision.models

from model.odas import ODAS
from torchsummary import summary
from tool import device
from tool.dataset import data_deal
from torch_lib import fit, evaluate
from model.loss import Loss
from tool.metrics import precision, recall, accuracy, f1, DSC, HM, IOU
from tool.callback import save_log
import time

train_dataset, test_dataset, neg_rate, _ = data_deal(batch_size=4, train_ratio=1.0, val_ratio=0.0)
print(torch.cuda.is_available())
model = ODAS().to(device)
model.load_state_dict(torch.load('model/saved_model/model.pt'))
# summary(model, (1, 512, 512))
# print(model)


def call(data):
    mode = data['model']
    for name, parms in mode.named_parameters():
        if name == 'fuse_layer.bias':
            print('-->name:', name, '-->grad_requirs:', parms, ' -->grad_value:', parms.grad)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss = Loss(weight=torch.tensor([1-neg_rate, neg_rate]).to(device))
fit(model=model, train_dataset=train_dataset, epochs=1000, metrics=[loss, accuracy, precision, recall, f1, DSC, HM, IOU], epoch_callbacks=[save_log], optimizer=optimizer, lr_decay='lambda', lr_decay_options={'lr_lambda': lambda epoch: (1-epoch / 1000) ** 0.9})
# eva = evaluate(model=model, dataset=test_dataset, metrics=[loss, accuracy, precision, recall, f1, DSC, HM, IOU])
# print(eva)
# with open('log/test_result.txt', 'w') as f:
#     time = time.asctime()
#     str1 = eva
#     f.write(str1)
