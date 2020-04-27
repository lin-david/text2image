# Text2image

Andrew Peng & David Lin

### Data

Download the Oxford 102 dataset [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) and the captions [here](https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view)

### Training

Update variable in `constants.py` and run `train.py`

### TODO in order of importance

1. How to get text embeddings
    - RNN (DONE)
    - Skip thought (DONE)
    - BERT (DONE)
    - GPT (DONE)
2. Losses (DONE)
3. Get it to train (DONE)
4. Put on GCP / AWS, add GPU support (DONE)
5. Benchmarks for accuracy (DONE)
6. Loading and saving models (DONE)
7. Generate image examples (DONE)
