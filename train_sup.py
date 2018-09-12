from gen import SineData
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from model.sup import Encoder, Decoder

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
        h = Variable(data['label'])

        z = enc(x)
        x_ = dec(F.max_pool2d(h, (1, 8)))

        loss1 = torch.sum((z.repeat([1, 1, 1, 8]) - h) ** 2)
        loss2 = torch.sum((x - x_) ** 2)

        loss = loss1 + loss2

        enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()
        enc_optim.step()
        dec_optim.step()

        if idx % 100 == 0:
            print(idx, loss)


if __name__ == '__main__':

    for epoch in range(1):
        train(epoch)

    import matplotlib.pyplot as plt


    def plot(idx):
        x = next(iter(dataloader))['wave']
        plt.plot(x.data.numpy()[idx, :])
        plt.show()

        z = enc(x.unsqueeze(1).unsqueeze(1))
        plt.imshow(z.data.numpy()[idx, :, :].reshape[9,:,:])
        plt.show()

        x_ = dec(z)
        plt.plot(x_.data.numpy()[idx, 0, 0, :])
        plt.show()


    plot(0)
    plot(1)
    plot(2)
