import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.ResNet8 import ResNet8

logger = logging.getLogger("logger")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator_MNIST(nn.Module):
    def __init__(self):
        super(Generator_MNIST, self).__init__()

        self.init_size = 28 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Generator_CIFAR(nn.Module):
    def __init__(self, the_dim=3):
        super(Generator_CIFAR, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, the_dim, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def gan_defense(d1, d2, params, the_epoch):
    lr = 0.0005
    b1 = 0.5
    b2 = 0.999
    latent_dim = 100
    classnum = 10
    image_num = 32
    discriminator1 = ResNet8(num_classes=classnum)
    discriminator2 = ResNet8(num_classes=classnum)
    dims = 3
    n_epochs1 = 10
    n_epochs2 = 5
    generator = Generator_CIFAR()

    discriminator1.load_state_dict(d1)
    discriminator2.load_state_dict(d2)

    generator.to(device)
    discriminator1.to(device)
    discriminator2.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    Tensor = torch.cuda.FloatTensor
    ori_imgs = []

    for c in range(classnum):
        para = None
        loss_min = 10000
        Alpha = 0.75
        generator.apply(weights_init_normal)
        for epoch in range(n_epochs1):
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (image_num, latent_dim))))
            gen_imgs = generator(z)
            out1 = torch.softmax(discriminator1(gen_imgs), dim=1)
            out1 = torch.std(out1, dim=1)
            out1 = torch.mean(out1)

            out2 = torch.softmax(discriminator2(gen_imgs), dim=1)
            out2 = out2[:, c]
            out2 = torch.mean(out2)
            g_loss = Alpha * out1 + (1 - Alpha) * (1 - out2)
            g_loss.backward()
            optimizer_G.step()

            if g_loss < loss_min:
                loss_min = g_loss
                para = generator.state_dict()

        generator.load_state_dict(para)
        generator.to(device)
        z = Variable(Tensor(np.random.normal(0, 1, (image_num, latent_dim))))
        imgs = generator(z)
        ori_imgs.append(imgs)

    if params['type'] != 'MNIST':
        generator = Generator_CIFAR(the_dim=1)
        generator = generator.to(device)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

    tri_imgs = []
    para = None
    loss_min = 10000
    generator.apply(weights_init_normal)
    generator.to(device)
    for epoch in range(n_epochs2):
        optimizer_G.zero_grad()
        z = Variable(Tensor(np.random.normal(0, 1, (image_num, latent_dim))))
        gen_trigger = generator(z)
        g_loss = 0
        other_imgs = torch.cat(ori_imgs[:0] + ori_imgs[1:])
        this_imgs = ori_imgs[0]
        new_gen_img = other_imgs.detach() + gen_trigger.repeat([classnum - 1, dims, 1, 1])
        out1 = torch.softmax(discriminator1(new_gen_img), dim=1)
        out1 = torch.std(out1, dim=1)
        out1 = torch.mean(out1, dim=0)

        out2 = torch.softmax(discriminator2(new_gen_img), dim=1)
        out2 = torch.mean(out2, dim=0)
        out2 = out2[0]

        g_loss = g_loss + 0.75 * out1 + (1 - 0.75) * (1 - out2)

        new_gen_trigger = gen_trigger.repeat([1, dims, 1, 1])
        new_gen_img = this_imgs.detach() - new_gen_trigger

        out1 = torch.softmax(discriminator1(new_gen_img), dim=1)
        out1 = torch.std(out1, dim=1)
        out1 = torch.mean(out1, dim=0)

        out2 = torch.softmax(discriminator2(new_gen_img), dim=1)
        out2 = torch.mean(out2, dim=0)
        out2 = out2[0]

        out3 = torch.softmax(discriminator1(new_gen_trigger), dim=1)
        out3 = torch.mean(out3, dim=0)
        out3 = out3[0]

        g_loss = g_loss + 0.75 * out1 + (1 - 0.75) * out2 + (1 - 0.75) * out3
        g_loss.backward()
        optimizer_G.step()
        if g_loss < loss_min:
            loss_min = g_loss
            para = generator.state_dict()
        generator.load_state_dict(para)
        generator.to(device)
        z = Variable(Tensor(np.random.normal(0, 1, (image_num, latent_dim))))
        with torch.no_grad():
            imgs = generator(z)
            tri_imgs.append(imgs.repeat([1, dims, 1, 1]))
    return tri_imgs
