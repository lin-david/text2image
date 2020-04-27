import os
import torch
import pickle
import argparse
import sys

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from data_pipeline import *
from constants import *
from model import *
from train import Trainer
from losses import gen_loss, disc_loss
from skipthoughts import *
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTModel

"""
Take from https://raw.githubusercontent.com/sbarratt/inception-score-pytorch/master/inception_score.py
and slightly modified
"""


def inception_score(imgs, model_file, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    device = torch.device('cuda' if cuda else 'cpu')

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, drop_last=True)

    # Load generator and embeddings
    if args.use_skip_thought:
        model = BayesianUniSkip('data/skip_thoughts', imgs.word_to_idx.keys())
        for param in model.parameters():
            param.requires_grad = False
    elif args.use_bert:
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
    elif args.use_gpt:
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        model.eval()
    else:
        model = RnnEncoder(dict_size=len(word_to_idx),
                         embed_size=args.embed_size,
                         hidden_dim=args.rnn_hidden_dim,
                         drop_prob=0.5)

    generator = Generator().to(device)
    trainer = Trainer(dataloader, model, generator, None, None, None, None, device, None)
    trainer.load_model(model_file)
    trainer.rnn_encoder.eval()
    trainer.generator.eval()

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N // batch_size, 1000))

    for i, batch in enumerate(dataloader, 0):
        print ("Calculating Inception Score... iter: {} / {}  ".format(i, N // batch_size), end='\r')
        # batch = batch.type(dtype)
        # batchv = Variable(batch)
        imgs, caps, cap_lens, fake_caps, fake_cap_lens = trainer.prepare_data(batch)

        # Text embedding
        sent_emb, fake_sent_emb = trainer.embed_text(caps, cap_lens, fake_caps, fake_cap_lens, batch_size)

        batch_size_i = caps.size()[0]
        sampled = torch.randn((batch_size_i, generator.z_size)).to(device)
        batchv = generator(sent_emb, sampled)
        
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)
    print()
    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--skip-thought', action='store_true')
    parser.add_argument('--use-bert', action='store_true')
    parser.add_argument('--use-gpt', action='store_true')
    parser.add_argument('--hidden-dim', type=int, default=2400)
    args = parser.parse_args()

    with open(FLOWERS_DATA_ROOT + 'idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    with open(FLOWERS_DATA_ROOT + 'word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = FlowerDataset(
        img_folder=FLOWERS_DATA_ROOT + 'jpg',
        text_folder=FLOWERS_DATA_ROOT + 'train_val',
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        transform=transform
    )

    cuda = True if torch.cuda.is_available() else False
    print("Using GPU: {}".format(cuda))
    print ("Calculating Inception Score...")
    print(inception_score(test_dataset, args.model_file, cuda=cuda, batch_size=16, resize=True))
