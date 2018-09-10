from gen import SineData
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model.ae import Encoder, Decoder

dataset = SineData()
dataloader = DataLoader(dataset, batch_size=10, num_workers=5)

enc = Encoder()
dec = Decoder()
enc_optim = torch.optim.Adam(enc.parameters(), lr=0.001)
dec_optim = torch.optim.Adam(dec.parameters(), lr=0.001)


def train(epoch):
    enc.train()
    dec.train()

    for idx, data in enumerate(dataloader):

        x = data['wave'].unsqueeze(1).unsqueeze(1)
        x = Variable(x)
        h = Variable(data['label'])

        z = enc(x)
        x_ = dec(z)

        loss = torch.sum((x - x_) ** 2)

        enc_optim.zero_grad()
        dec_optim.zero_grad()
        loss.backward()
        enc_optim.step()
        dec_optim.step()

        if idx % 100 == 0:
            print(idx, loss)


if __name__ == '__main__':

    for epoch in range(3):
        train(epoch)

    import matplotlib.pyplot as plt


    def plot(idx):
        a = next(iter(dataloader))
        plt.plot(a.data.numpy()[idx, :])
        plt.show()

        b = enc(a.unsqueeze(1).unsqueeze(1))
        plt.imshow(b.data.numpy()[idx, 0, :])
        plt.show()

        c = dec(b)
        plt.plot(c.data.numpy()[idx, 0, 0, :])
        plt.show()


    plot(0)
    plot(1)
    plot(2)
