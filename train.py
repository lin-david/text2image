import os
import pickle
import argparse
import sys

sys.path.append('skip-thoughts.torch/pytorch')
import torch
from torch.utils.data import TensorDataset

from data_pipeline import *
from constants import *
from model import *
from losses import gen_loss, disc_loss

import matplotlib.pyplot as plt
from torchvision import transforms, utils
from skipthoughts import *
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTModel, OpenAIGPTPreTrainedModel


class Trainer():

    def __init__(self, dataset, rnn_encoder, generator, discriminator, optim_g, optim_d, num_epochs, device, out_img_folder, out_model_file='models/', save_every=50, *args, **kwargs):
        self.dataset = dataset
        self.rnn_encoder = rnn_encoder
        self.generator = generator
        self.discriminator = discriminator
        self.optim_gen = optim_g
        self.optim_disc = optim_d
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.device = device
        self.out_img_folder = out_img_folder
        self.out_model_file = out_model_file

        # Uncomment this line to load an existing model
        # self.load_model()

    def save_model(self, path='models/', added=''):
        torch.save(self.rnn_encoder.state_dict(), path+'rnn_encoder{}.pt'.format(added))
        torch.save(self.generator.state_dict(), path+'generator{}.pt'.format(added))
        torch.save(self.discriminator.state_dict(), path+'discriminator{}.pt'.format(added))

    def load_model(self, path='models/', added=''):
        self.rnn_encoder.load_state_dict(torch.load(path+'rnn_encoder.pt', map_location=self.device))
        self.generator.load_state_dict(torch.load(path+'generator.pt', map_location=self.device))
        self.discriminator.load_state_dict(torch.load(path+'discriminator.pt', map_location=self.device))

        self.rnn_encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

    def prepare_data(self, datum):
        real_lens = datum['real_len'].to(self.device)
        fake_lens = datum['fake_len'].to(self.device)
        images = datum['image'].to(self.device)
        real_cap = datum['real'].to(self.device)
        fake_cap = datum['fake'].to(self.device)

        sorted_cap_lens, sorted_cap_indices = torch.sort(real_lens, 0, True)
        captions = real_cap[sorted_cap_indices].squeeze()
        images = images[sorted_cap_indices]

        sorted_fake_lens, sorted_fake_cap_indices = torch.sort(fake_lens, 0, True)
        fake_captions = fake_cap[sorted_fake_cap_indices].squeeze()

        if isinstance(self.rnn_encoder, BertPreTrainedModel):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

            # Captions
            captions = captions.tolist()
            new_captions = []
            new_cap_mask = []

            # Fake captions
            fake_captions = fake_captions.tolist()
            new_fake_captions = []
            new_fake_cap_mask = []

            for i in range(len(captions)):
                # Captions
                text = [self.dataset.dataset.idx_to_word[w] for w in captions[i]][:sorted_cap_lens[i]]

                tokenized_text = []
                tokenized_text.append("[CLS]")
                tokenized_text.extend(tokenizer.tokenize(' '.join(text)))
                tokenized_text.append("[SEP]")

                input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_mask = [1] * len(input_ids)

                diff = (train_val_dataset.max_len + 8) - len(input_ids)
                input_ids.extend([0] * diff)
                input_mask.extend([0] * diff)

                new_captions.append(input_ids)
                new_cap_mask.append(input_mask)

                # Fake captions
                fake_text = [self.dataset.dataset.idx_to_word[w] for w in fake_captions[i]][:sorted_fake_lens[i]]

                fake_tokenized_text = []
                fake_tokenized_text.append("[CLS]")
                fake_tokenized_text.extend(tokenizer.tokenize(' '.join(fake_text)))
                fake_tokenized_text.append("[SEP]")

                fake_input_ids = tokenizer.convert_tokens_to_ids(fake_tokenized_text)
                fake_input_mask = [1] * len(fake_input_ids)

                diff = (train_val_dataset.max_len + 8) - len(fake_input_ids)
                fake_input_ids.extend([0] * diff)
                fake_input_mask.extend([0] * diff)

                new_fake_captions.append(fake_input_ids)
                new_fake_cap_mask.append(fake_input_mask)

            captions = torch.LongTensor(new_captions)
            sorted_cap_lens = torch.LongTensor(new_cap_mask)

            fake_captions = torch.LongTensor(new_fake_captions)
            sorted_fake_lens = torch.LongTensor(new_fake_cap_mask)

        return images, captions, sorted_cap_lens, fake_captions, sorted_fake_lens

    def embed_text(self, caps, cap_lens, fake_caps, fake_cap_lens, batch_size):
        if isinstance(self.rnn_encoder, AbstractSkipThoughts):
            sent_emb = self.rnn_encoder(caps.long(), lengths=cap_lens.cpu().numpy().tolist())
            fake_sent_emb = self.rnn_encoder(fake_caps.long(), lengths=fake_cap_lens.cpu().numpy().tolist())
        elif isinstance(self.rnn_encoder, BertPreTrainedModel):
            cap_segments = torch.zeros(tuple(caps.size()), dtype=torch.long).to(self.device)
            _, sent_emb = self.rnn_encoder(caps.to(self.device),
                                           token_type_ids=cap_segments,
                                           attention_mask=cap_lens.to(self.device),
                                           output_all_encoded_layers=False)

            fake_cap_segments = torch.zeros(tuple(fake_caps.size()), dtype=torch.long).to(self.device)
            _, fake_sent_emb = self.rnn_encoder(fake_caps.long().to(self.device),
                                                token_type_ids=fake_cap_segments,
                                                attention_mask=fake_cap_lens.to(self.device),
                                                output_all_encoded_layers=False)
        elif isinstance(self.rnn_encoder, OpenAIGPTPreTrainedModel):
            sent_emb = self.rnn_encoder(caps.long())
            sent_emb = torch.sum(sent_emb, dim=1)
            fake_sent_emb = self.rnn_encoder(fake_caps.long())
            fake_sent_emb = torch.sum(fake_sent_emb, dim=1)
        else:
            init_hidden = self.rnn_encoder.init_hidden(batch_size).to(self.device).detach()
            _, sent_emb = self.rnn_encoder(caps, cap_lens, init_hidden)

            init_fake_hidden = self.rnn_encoder.init_hidden(batch_size).to(self.device).detach()
            _, fake_sent_emb = self.rnn_encoder(fake_caps, fake_cap_lens, init_fake_hidden)

        return sent_emb, fake_sent_emb

    def train(self):
        self.rnn_encoder.train()
        self.generator.train()
        self.discriminator.train()

        d_losses = []
        g_losses = []
        real_real_losses = []
        fake_real_losses = []
        real_fake_losses = []
        class_losses = []
        feat_losses = []
        dist_losses = []

        for i in range(self.num_epochs):
            total_g_loss = 0
            total_d_loss = 0
            for j, d in enumerate(self.dataset):
                imgs, caps, cap_lens, fake_caps, fake_cap_lens = self.prepare_data(d)
                batch_size = imgs.size(0)

                self.discriminator.zero_grad()
                self.rnn_encoder.zero_grad()

                # Text embedding
                sent_emb, fake_sent_emb = self.embed_text(caps, cap_lens, fake_caps, fake_cap_lens, batch_size)

                # Generate image
                sampled = torch.randn((batch_size, self.generator.z_size)).to(self.device)
                fake_imgs = self.generator(sent_emb.detach(), sampled)

                # Update discriminator
                real_real, fake_real, real_fake, d_loss = disc_loss(self.discriminator, imgs, fake_imgs, sent_emb.detach(), fake_sent_emb.detach(), self.device)
                d_loss.backward()
                self.optim_disc.step()

                # Update generator
                self.rnn_encoder.zero_grad()
                self.generator.zero_grad()
                sampled = torch.randn((batch_size, self.generator.z_size)).to(self.device)
                fake_imgs = self.generator(sent_emb, sampled)
                class_loss, feat_loss, dist_loss, g_loss = gen_loss(self.discriminator, imgs, fake_imgs, sent_emb, self.device)
                g_loss.backward()
                self.optim_gen.step()

                # Logging
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_real_losses.append(real_real)
                fake_real_losses.append(fake_real)
                real_fake_losses.append(real_fake)
                class_losses.append(class_loss)
                feat_losses.append(feat_loss)
                dist_losses.append(dist_loss)

                if j != 0 and j % self.save_every == 0:
                    self.save_image(fake_imgs[0], caps[0], "{}_{}".format(i, j))
                    print("Epoch {} iter {} - D Loss: {} / G Loss: {}".format(i, j, fake_real.item(), class_loss.item()))
            print("===============================")
            print("Epoch {} D loss: {}".format(i, total_d_loss))
            print("Epoch {} G loss: {}".format(i, total_g_loss))
            print("===============================")
            if i > 0 and i % 20 == 0:
                self.save_model(self.out_model_file, i)
        fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, sharex=True)
        axs1.plot(d_losses)
        axs2.plot(real_real_losses)
        axs3.plot(fake_real_losses)
        axs4.plot(real_fake_losses)
        fig.suptitle('Discriminator Loss', fontsize=16)
        axs1.set_title('Total loss')
        axs1.set_xlabel('Iterations')
        axs1.set_ylabel('Loss')
        axs2.set_title('Real image Real caption')
        axs2.set_xlabel('Iterations')
        axs2.set_ylabel('Loss')
        axs3.set_title('Fake image Real caption')
        axs3.set_xlabel('Iterations')
        axs3.set_ylabel('Loss')
        axs4.set_title('Real image Fake caption')
        axs4.set_xlabel('Iterations')
        axs4.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig("{}/{}.png".format(self.out_img_folder, "disc_loss"))
#         plt.show()
        plt.close(fig)

        fig, ((axs1, axs2), (axs3, axs4)) = plt.subplots(2, 2, sharex=True)
        axs1.plot(g_losses)
        axs2.plot(class_losses)
        axs3.plot(feat_losses)
        axs4.plot(dist_losses)
        fig.suptitle('Generator Loss', fontsize=16)
        axs1.set_title('Total loss')
        axs1.set_xlabel('Iterations')
        axs1.set_ylabel('Loss')
        axs2.set_title('BCE loss')
        axs2.set_xlabel('Iterations')
        axs2.set_ylabel('Loss')
        axs3.set_title('L2 feature loss')
        axs3.set_xlabel('Iterations')
        axs3.set_ylabel('Loss')
        axs4.set_title('L1 image loss')
        axs4.set_xlabel('Iterations')
        axs4.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig("{}/{}.png".format(self.out_img_folder, "gen_loss"))
#         plt.show()
        plt.close(fig)

    def generate(self, text, output_file, count=5):
        self.rnn_encoder.eval()
        self.generator.eval()

        tokens = tokenize(text)
        tokens = [self.dataset.dataset.word_to_idx[i] for i in tokens]
        len_tensor = torch.Tensor([len(tokens)]).int().to(self.device)
        token_tensor = torch.Tensor(tokens).view(1, len(tokens)).long().to(self.device)

        fake_imgs = []
        trans = transforms.ToPILImage()

        for i in range(count):
            if isinstance(self.rnn_encoder, AbstractSkipThoughts):
                sent_emb = self.rnn_encoder(token_tensor, lengths=[len(tokens)])
            elif isinstance(self.rnn_encoder, BertPreTrainedModel):
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                token_tensor = []
                token_tensor_mask = []

                tokenized_text = []
                tokenized_text.append("[CLS]")
                tokenized_text.extend(tokenizer.tokenize(text))
                tokenized_text.append("[SEP]")
                input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
                input_mask = [1] * len(input_ids)
                diff = 74 - len(input_ids)
                input_ids.extend([0] * diff)
                input_mask.extend([0] * diff)
                token_tensor.append(input_ids)
                token_tensor_mask.append(input_mask)

                token_tensor = torch.LongTensor(token_tensor)
                token_tensor_mask = torch.LongTensor(token_tensor_mask)

                cap_segments = torch.zeros(tuple(token_tensor.size()), dtype=torch.long).to(self.device)
                _, sent_emb = self.rnn_encoder(token_tensor.to(self.device),
                                               token_type_ids=cap_segments,
                                               attention_mask=token_tensor_mask.to(self.device),
                                               output_all_encoded_layers=False)
            elif isinstance(self.rnn_encoder, OpenAIGPTPreTrainedModel):
                sent_emb = self.rnn_encoder(token_tensor.long())
            else:
                init_hidden = self.rnn_encoder.init_hidden(1).to(self.device)
                _, sent_emb = self.rnn_encoder(token_tensor, len_tensor, init_hidden)
            sampled = torch.randn((1, self.generator.z_size)).to(device)
            fake_img = self.generator(sent_emb, sampled).squeeze(0) * 0.5 + 0.5
            fake_imgs.append(fake_img)
            # trans(fake_img.detach().cpu()).save("{}_{}".format(i, output_file))

        fake_imgs = [i.cpu().detach().numpy() for i in fake_imgs]
        _, axs = plt.subplots(1, count, figsize=(12, 12))
        axs = axs.flatten()

        for img, ax in zip(fake_imgs, axs):
            ax.imshow(img.transpose(1, 2, 0))

        plt.show()

    def interpolate(self, text1, text2, output_folder):
        self.rnn_encoder.eval()
        self.generator.eval()

        tokens_1 = tokenize(text1)
        tokens_1 = [self.dataset.dataset.word_to_idx[i] for i in tokens_1]

        tokens_2 = tokenize(text2)
        tokens_2 = [self.dataset.dataset.word_to_idx[i] for i in tokens_2]

        len_tensor_1 = torch.Tensor([len(tokens_1)]).int().to(self.device)
        token_tensor_1 = torch.Tensor(tokens_1).view(1, len(tokens_1)).long().to(self.device)

        len_tensor_2 = torch.Tensor([len(tokens_2)]).int().to(self.device)
        token_tensor_2 = torch.Tensor(tokens_2).view(1, len(tokens_2)).long().to(self.device)

        sent_emb_1, sent_emb_2 = self.embed_text(token_tensor_1, len_tensor_1, token_tensor_2, len_tensor_2, 1)

        weights = np.arange(0.1, 1, 0.1)

        # Fix random z
        sampled = torch.randn((1, self.generator.z_size)).to(device)
        tensors = [sent_emb_2]
        for w in weights:
            tensors.append(w * sent_emb_1 + (1 - w) * sent_emb_2)
        tensors.append(sent_emb_1)
        sampled = torch.stack([sampled for _ in range(len(tensors))]).squeeze(1)
        tensors = torch.stack(tensors).squeeze(1)
        fake_imgs = self.generator(tensors, sampled) * 0.5 + 0.5
        trans = transforms.ToPILImage()
        for i in range(fake_imgs.size(0)):
            fake_img = trans(fake_imgs[i].detach().cpu()).save(os.path.join(output_folder, '{}.jpg'.format(i)))
        return fake_imgs

    def save_image(self, image, caption, name):
        trans = transforms.ToPILImage()
        image = image.clone() * 0.5 + 0.5
        trans(image.detach().cpu()).save("{}/{}.jpg".format(self.out_img_folder, name))
        caption = caption.clone().detach().cpu().numpy()

        with open("{}/{}".format(self.out_img_folder, name), 'w') as f:
            if args.use_bert:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                f.write(" ".join(tokenizer.convert_ids_to_tokens(caption)))
            else:
                f.write(" ".join([self.dataset.dataset.idx_to_word[w] for w in caption]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='learning rate (dafault: 0.0002)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='beta1 for Adam optimizer (dafault: 0.5)')
    parser.add_argument('--use-skip-thought', action='store_true',
                        help='use pretrained skip thought embedding')
    parser.add_argument('--use-bert', action='store_true',
                        help='use Bidirectional Encoder Representations from Transformers')
    parser.add_argument('--use-gpt', action='store_true',
                        help='use OpenAI Generative Pre-Training')
    parser.add_argument('--output-model-file', type=str, default='model/',
                        help='folder to store model files')
    parser.add_argument('--output-image-folder', type=str, default='generated_images',
                        help='folder to store image files')
    parser.add_argument('--rnn-hidden-dim', type=int, default=128)
    parser.add_argument('--embed-size', type=int, default=300)
    args = parser.parse_args()

    if not os.path.isdir(FLOWERS_DATA_ROOT + 'train_val'):
        print("Grouping data")
        num_train, num_test = group_data(
            TRAINVAL_CF,
            TEST_CF,
            TEXT_F,
        )
        print("%d number of train + val classes" % num_train)
        print("%d number of test classes" % num_test)

    # Random Crop
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(FLOWERS_DATA_ROOT + 'idx_to_word.pkl'):
        idx_to_word, word_to_idx = build_dictionary([FLOWERS_DATA_ROOT + 'train_val', FLOWERS_DATA_ROOT + 'test'])
        with open(FLOWERS_DATA_ROOT + 'idx_to_word.pkl', 'wb') as f:
            pickle.dump(idx_to_word, f)
        with open(FLOWERS_DATA_ROOT + 'word_to_idx.pkl', 'wb') as f:
            pickle.dump(word_to_idx, f)
    else:
        with open(FLOWERS_DATA_ROOT + 'idx_to_word.pkl', 'rb') as f:
            idx_to_word = pickle.load(f)
        with open(FLOWERS_DATA_ROOT + 'word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)

    train_val_dataset = FlowerDataset(
        img_folder=FLOWERS_DATA_ROOT + 'jpg',
        text_folder=FLOWERS_DATA_ROOT + 'train_val',
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(train_val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             drop_last=True)

    if args.use_skip_thought:
        model = BayesianUniSkip('data/skip_thoughts', word_to_idx.keys())
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

    generator = Generator()
    discriminator = Discriminator()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}".format(device))

    model = model.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optim_gen = torch.optim.Adam(filter(lambda x: x.requires_grad,
                                        list(generator.parameters()) + list(model.parameters())),
                                 lr=args.learning_rate,
                                 betas=(args.momentum, 0.999))
    optim_disc = torch.optim.Adam(filter(lambda x: x.requires_grad, list(discriminator.parameters())),
                                 lr=args.learning_rate,
                                 betas=(args.momentum, 0.999))
    trainer = Trainer(dataloader, model, generator, discriminator, optim_gen, optim_disc, args.epochs, device,
                      args.output_image_folder, args.output_model_file)
    trainer.train()
    trainer.save_model(path=args.output_model_file)
