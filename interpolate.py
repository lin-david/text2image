import os
import sys
import pickle
import argparse
import torch
from torchvision import transforms, utils

sys.path.append('skip-thoughts.torch/pytorch')

from constants import *
from data_pipeline import *
from model import *
from skipthoughts import *
from train import Trainer

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTModel, OpenAIGPTPreTrainedModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-folder', type=str, required=True,
                        help='Output folder name')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model files')
    parser.add_argument('--text1', type=str, required=True)
    parser.add_argument('--text2', type=str, required=True)
    parser.add_argument('--use-skip-thought', action='store_true',
                        help='use pretrained skip thought embedding')
    parser.add_argument('--use-bert', action='store_true',
                        help='use pretrained BERT embedding')
    parser.add_argument('--use-gpt', action='store_true',
                        help='use pretrained GPT embedding')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                        help='RNN hidden dim size')
    parser.add_argument('--embed-size', type=int, default=1024,
                        help='Word embed size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}".format(device))

    with open(FLOWERS_DATA_ROOT + 'idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    with open(FLOWERS_DATA_ROOT + 'word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    train_val_dataset = FlowerDataset(
        img_folder=FLOWERS_DATA_ROOT + 'jpg',
        text_folder=FLOWERS_DATA_ROOT + 'train_val',
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
    )

    if args.use_skip_thought:
        model = BayesianUniSkip('data/skip_thoughts', word_to_idx.keys())
    elif args.use_bert:
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
    elif args.use_gpt:
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        model.eval()
    else:
        model = RnnEncoder(dict_size=len(word_to_idx),
                         embed_size=args.embed_size,
                         hidden_dim=args.hidden_dim,
                         drop_prob=0.5)
    generator = Generator()
    discriminator = Discriminator()

    dataloader = torch.utils.data.DataLoader(train_val_dataset,
                                             batch_size=1,
                                             shuffle=True)
    model = model.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    trainer = Trainer(dataloader, model, generator, discriminator, None, None, 1, device, None)
    print("Loading model files")
    trainer.load_model(args.model_path)
    print("Generating image from text")
    trainer.interpolate(args.text1, args.text2, args.output_folder)
    print("Images saved to {}".format(args.output_folder))
