import os
import glob
import torch
import nltk
import random
import shutil

from collections import Counter
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from constants import FLOWERS_DATA_ROOT

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def tokenize(line):
    line = line.rstrip().lower()
    tokenizer = nltk.tokenize.regexp.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(line.lower())
    return tokens


def group_data(trainval_cf, test_cf, class_root):
    """
    Args:
        train_cf: File name for train classes
        valid_cf: File name for valid_classes
        test_cf: File name for test_classes
        text_folder: Folder name for image descriptions
        class_root: Folder where classes are located

    Returns:
        length  of train/valid, test sets
    """
    with open(trainval_cf, 'r') as f:
        trainval_classes = f.read().splitlines()

    with open(test_cf, 'r') as f:
        test_classes = f.read().splitlines()

    for folder in ['train_val', 'test']:
        if not os.path.isdir(FLOWERS_DATA_ROOT + folder):
            os.makedirs(FLOWERS_DATA_ROOT + folder)

    for classes, name in zip([trainval_classes, test_classes], ['train_val', 'test']):
        for c in classes:
            files = glob.glob('/'.join([class_root, c, '*.txt']))
            for f in files:
                shutil.copy(f, FLOWERS_DATA_ROOT + name)

    return len(trainval_classes), len(test_classes)


def build_dictionary(caption_folders):
    files = []
    for folder in caption_folders:
        files.extend(glob.glob('/'.join([folder, '*.txt'])))
    captions = []
    for cur_f in files:
        with open(cur_f, 'r') as f:
            count = 0
            for line in f:
                caption = tokenize(line)
                captions.append(caption)
                count += 1
            assert count == 10, "There should be 10 captions per image"

    word_counts = Counter()
    for cap in captions:
        for word in cap:
            word_counts[word] += 1

    vocab = word_counts.keys()
    idx_to_word = {}
    idx_to_word[0] = '<start>'
    idx_to_word[1] = '<end>'
    idx_to_word[2] = '<pad>'

    word_to_idx = {}
    word_to_idx['<start>'] = 0
    word_to_idx['<end>'] = 1
    word_to_idx['<pad>'] = 2

    for i, w in enumerate(vocab, start=3):
        idx_to_word[i] = w
        word_to_idx[w] = i

    return idx_to_word, word_to_idx


class FlowerDataset(Dataset):

    def __init__(self, img_folder, text_folder, word_to_idx=None, idx_to_word=None, transform=None):
        self.img_folder = img_folder
        self.text_folder = text_folder 
        self.transforms = transforms
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.transform = transform
        self.max_len = 0
        self.list_id = self._map_idx()
        self.captions = self._load_captions()

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, idx):
        img_str = self.list_id[idx].replace('.txt', '.jpg')
        img_path = os.path.join(self.img_folder, img_str)
        image = Image.open(img_path)
        captions = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        # select random caption
        cap = np.array(random.choice(captions))
        fake_cap_i = random.randint(0, len(self.captions) - 1)
        fake_cap = np.array(random.choice(self.captions[fake_cap_i]))
        cap_len = len(cap)
        fake_cap_len = len(fake_cap)
        cap = np.pad(cap, (0, self.max_len - len(cap)), 'constant', constant_values=2)
        fake_cap = np.pad(fake_cap, (0, self.max_len - len(fake_cap)), 'constant', constant_values=2)
        sample = {
            'image': image,
            'real': torch.Tensor(cap),
            'fake': torch.Tensor(fake_cap),
            'real_len': cap_len,
            'fake_len': fake_cap_len
        }
        return sample

    def _map_idx(self):
        ids = {}
        count = 0
        for f in os.listdir(self.text_folder):
            if os.path.isfile(os.path.join(self.text_folder, f)):
                ids[count] = f
                count += 1
        return ids

    def _load_captions(self):
        idx_to_caps = {}
        for i, txt_str in self.list_id.items():
            caption_path = os.path.join(self.text_folder, txt_str)
            with open(caption_path, 'r') as f:
                idx_to_caps[i] = []
                for line in f:
                    caption = tokenize(line)
                    if len(caption) == 0:
                        continue
                    idx_to_caps[i].append([self.word_to_idx[w] for w in caption])
                    self.max_len = max(self.max_len, len(caption))
                assert len(idx_to_caps[i]) > 0, "At least 1 caption per image"
        return idx_to_caps
    
    def assert_cap_len(self, length):
        for i, captions in self.captions.items():
            for cap in captions:
                pad_cap = np.pad(cap, (0, self.max_len - len(cap)), 'constant', constant_values=2) 
                assert len(pad_cap) == length, "{}_{}".format(i, " ".join([self.idx_to_word[i] for i in cap]))
