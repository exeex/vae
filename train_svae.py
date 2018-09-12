from gen import SineData
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from model.svae import Encoder, Decoder

dataset = SineData()
dataloader = DataLoader(dataset, batch_size=10, num_workers=5)

enc = Encoder()
dec = Decoder()
enc_optim = torch.optim.Adam(enc.parameters(), lr=0.001)
dec_optim = torch.optim.Adam(dec.parameters(), lr=0.001)


# TODO: verify this, and implement vae.

def train(epoch):
    enc.train()
    dec.train()

    for idx, data in enumerate(dataloader):

        x = data['wave'].unsqueeze(1).unsqueeze(1)
        x = Variable(x)
        label = Variable(data['label'])

        label_, phase = enc(x)

        # print(label_.shape, phase.shape)

        x_ = dec(F.avg_pool2d(label_, (1, 8)), phase)

        loss1 = torch.sum((label_ - F.avg_pool2d(label, (1, 2))) ** 2)
        loss2 = torch.sum((x_ - F.avg_pool2d(x, (1, 2))) ** 2)

        loss = loss1 + loss2

        enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()
        enc_optim.step()
        dec_optim.step()

        if idx % 100 == 0:
            print(idx, loss1, loss2)


for epoch in range(3):
    train(epoch)

import matplotlib.pyplot as plt


def plot(idx):
    x = next(iter(dataloader))['wave']
    plt.plot(x.data.numpy()[idx, :])
    plt.show()

    z_l, z_p = enc(x.unsqueeze(1).unsqueeze(1))
    img = z_l.data.numpy()[idx, :, :]
    print(img.shape)
    plt.imshow(img.reshape(1, 9, 256)[0, :, :])
    plt.show()

    x_ = dec(F.max_pool2d(z_l, (1, 8)), z_p)
    plt.plot(x_.data.numpy()[idx, 0, 0, :])
    plt.show()


plot(0)
plot(1)
plot(2)
