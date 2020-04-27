import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import TEXT_FEATURE_SIZE, TEXT_EMBEDDING_SIZE

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RnnEncoder(nn.Module):
    """
    See https://github.com/taoxugit/AttnGAN/blob/master/code/model.py:RNN_ENCODER
    """
    def __init__(self, dict_size, embed_size, drop_prob, hidden_dim=128, layers=1, bidirectional=True):
        super(RnnEncoder, self).__init__()
        self.dict_size = dict_size
        self.embed_size = embed_size
        self.drop_prob = drop_prob
        self.hidden_dim = hidden_dim if not bidirectional else hidden_dim // 2
        self.layers = layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(self.dict_size, self.embed_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.rnn = nn.GRU(self.embed_size,
                          self.hidden_dim,
                          self.layers,
                          batch_first=True,
                          dropout=self.drop_prob,
                          bidirectional=self.bidirectional)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.embedding = self.embedding.to(*args, **kwargs) 
        return self

    def init_hidden(self, batch_size):
        layers = 2 * self.layers if self.bidirectional else self.layers
        return torch.zeros((layers, batch_size, self.hidden_dim))

    def forward(self, captions, cap_len, hidden, mask=None):
        captions = captions.long()
        embeddings = self.dropout(self.embedding(captions))
        cap_len = cap_len.data.tolist()

        # sorted cap len
        embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, cap_len, batch_first=True)
        out, hidden = self.rnn(embeddings, hidden)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        out = out.transpose(1, 2)
        sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.hidden_dim if not self.bidirectional else self.hidden_dim * 2)
        return out, sent_emb


class Generator(nn.Module):

    def __init__(self, z_size=100, filter_base=64, inplace=True):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.deconv = nn.Sequential(
            # Latent vector -> filter * 8 x 4 x 4
            nn.ConvTranspose2d(self.z_size + TEXT_EMBEDDING_SIZE, filter_base * 8, 4, bias=False),
            # nn.BatchNorm2d(filter_base * 8),
            nn.ReLU(inplace),
            # filter * 8 x 4 x 4 -> filter * 4 x 8 x 8
            nn.ConvTranspose2d(filter_base * 8, filter_base * 4, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(filter_base * 4),
            nn.ReLU(inplace),
            # filter * 4 x 8 x 8 -> filter * 2 x 16 x 16
            nn.ConvTranspose2d(filter_base * 4, filter_base * 2, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(filter_base * 2),
            nn.ReLU(inplace),
            # filter * 2 x 16 x 16 -> filter x 32 x 32
            nn.ConvTranspose2d(filter_base * 2, filter_base, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(filter_base),
            nn.ReLU(inplace),
            # filter x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(filter_base, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.projection = nn.Sequential(
            nn.Linear(TEXT_FEATURE_SIZE, TEXT_EMBEDDING_SIZE),
            # nn.BatchNorm1d(num_features=TEXT_EMBEDDING_SIZE),
            nn.LeakyReLU(0.2, inplace))

    def forward(self, x, sampled):
        x = self.projection(x)
        input_vect = torch.cat((sampled, x), dim=1)
        input_vect = input_vect.unsqueeze(-1).unsqueeze(-1)
        image = self.deconv(input_vect)
        return image


class Discriminator(nn.Module):

    def __init__(self, filter_base=64, inplace=True):
        super(Discriminator, self).__init__()
        self.forward_conv = nn.Sequential(
            # 3 x 64 x 64 -> filter x 32 x 32
            nn.Conv2d(3, filter_base, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_base),
            nn.LeakyReLU(0.2, inplace),
            # filter x 32 x 32 -> 2 * filter x 16 x 16
            nn.Conv2d(filter_base, filter_base * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_base * 2),
            nn.LeakyReLU(0.2, inplace),
            # 2 * filter x 16 x 16 -> 4 * filter x 8 x 8
            nn.Conv2d(filter_base * 2, filter_base * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_base * 4),
            nn.LeakyReLU(0.2, inplace),
            # 4 * filter x 8 x 8 -> 8 * filter x 4 x 4
            nn.Conv2d(filter_base * 4, filter_base * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filter_base * 8),
            nn.LeakyReLU(0.2, inplace),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(filter_base * 8 + TEXT_EMBEDDING_SIZE, filter_base * 8, 1, bias=False),
            nn.BatchNorm2d(filter_base * 8),
            nn.LeakyReLU(0.2, inplace),
            nn.Conv2d(filter_base * 8, 1, 4)
        )
        self.projection = nn.Linear(TEXT_FEATURE_SIZE, TEXT_EMBEDDING_SIZE)

    def forward(self, x, latent):
        latent = self.projection(latent)
        features = self.forward_conv(x)

        latent = latent.unsqueeze(-1).unsqueeze(-1)
        latent = latent.repeat(1, 1, features.size(2), features.size(3))

        combined = torch.cat((features, latent), dim=1)
        output = self.classifier(combined)
        output = output.view(output.size(0))
        return output, features


