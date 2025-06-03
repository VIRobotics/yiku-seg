import torch
from torch import nn
from .Block import Encoder_aae,Decoder,Discriminator
class AAE(nn.Module):
    def __init__(self,img_sz,latent):
        super(AAE,self).__init__()
        self.ch=3
        self.imgsz=img_sz
        self.latent=latent
        self._enc=Encoder_aae(latent)
        self._dec=Decoder(img_sz, latent)
        self._discriminator=Discriminator(latent)
        self.real_label = torch.ones([self.opt.batch_size, 1])
        self.fake_label = torch.zeros([self.opt.batch_size, 1])

    def forward(self,x):
        x = self._enc(x)
        return self._dec(x)

    def forward_dm(self,x):
        z_fake = self._enc(x)
        self.fake = self.discriminator(z_fake)
        z_real_gauss = torch.randn(self.real_imgs.size()[0], self.opt.latent)
        self.real = self.discriminator(z_real_gauss)

        self.real_label = self.real_label.type_as(self.real)
        self.fake_label = self.fake_label.type_as(self.fake)

    def backward_recon(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.eval()
        self.recon_loss = self.criterion(10. * self.real_imgs, 10. * self.generated_imgs)
        self.recon_loss.backward()

    def backward_dm(self):
        # discriminator train
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()
        self.dm_loss = self.criterion_dm(self.real, self.real_label) + self.criterion_dm(self.fake, self.fake_label)
        self.dm_loss.backward()

    def backward_g(self):
        # generator train
        self.encoder.train()
        self.discriminator.eval()
        self.fake = self.discriminator(self.encoder(self.real_imgs))
        self.g_loss = self.criterion_dm(self.fake, self.real_label)
        self.g_loss.backward()

