import os
import zipfile

TXT_EMBS_PATH = '/data/dataset/tiny_imagenet/txt_embeddings.zip'
with zipfile.ZipFile(TXT_EMBS_PATH, 'r') as zip_ref:
    zip_ref.extractall(TXT_EMBS_PATH[:-4])

