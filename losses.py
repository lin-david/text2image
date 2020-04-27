import torch
import torch.nn as nn
import torch.nn.functional as F


def disc_loss(discriminator, real_imgs, fake_imgs, real_embed, fake_embed, device):
    jitter_mean = torch.zeros(real_imgs.size()).to(device)
    jitter_std = torch.ones(real_imgs.size()).to(device) * 0.05
    jitter = torch.normal(jitter_mean, jitter_std)
    real_real, _ = discriminator(real_imgs + jitter, real_embed)
    fake_real, _ = discriminator(fake_imgs + jitter, real_embed)
    real_fake, _ = discriminator(real_imgs + jitter, fake_embed)

    ones = torch.ones(real_imgs.size(0)).to(device)
    zeros = torch.zeros(real_imgs.size(0)).to(device)

    # Smooth label
    ones = torch.FloatTensor(real_imgs.size(0)).uniform_(0.95, 1).to(device)

    loss_real_real = F.binary_cross_entropy_with_logits(real_real, ones)
    loss_fake_real = F.binary_cross_entropy_with_logits(fake_real, zeros)
    loss_real_fake = F.binary_cross_entropy_with_logits(real_fake, zeros)

    total_loss = loss_real_real + loss_fake_real + loss_real_fake
    return loss_real_real, loss_fake_real, loss_real_fake, total_loss


def gen_loss(discriminator, real_imgs, fake_imgs, real_embed, device, l1_coef=0, l2_coef=0.5):
    logits, activation_fake = discriminator(fake_imgs, real_embed)
    _, activation_real = discriminator(real_imgs, real_embed)
    ones = torch.ones(fake_imgs.size(0)).to(device)

    loss = F.binary_cross_entropy_with_logits(logits, ones)
    feature_loss = l2_coef * F.mse_loss(activation_real.detach(), activation_fake)
    dist_loss = l1_coef * F.l1_loss(real_imgs, fake_imgs)
    total_loss = loss + feature_loss + dist_loss
    return loss, feature_loss, dist_loss, total_loss
