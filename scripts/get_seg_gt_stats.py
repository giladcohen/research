import numpy as np
import os
import matplotlib.pyplot as plt
import mmcv
import imageio

IMG_PATH = '/data/dataset/VOCdevkit/VOC2012/JPEGImages'
ANN_DIR1 = '/data/dataset/VOCdevkit/VOC2012/SegmentationClass'
ANN_DIR2 = '/data/dataset/VOCdevkit/VOC2012/SegmentationClassAug'
ANN_FILES1 = '/data/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
ANN_FILES2 = '/data/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/aug.txt'

values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 255]
cnt = {}
for i in values:
    cnt[i] = 0


with open(ANN_FILES1, "r") as f:
    content = f.read()
files = content.split("\n")
files.remove('')
for file in files:
    img_file = os.path.join(IMG_PATH, file + '.jpg')
    ann_file = os.path.join(ANN_DIR1, file + '.png')
    assert os.path.exists(img_file), img_file + ' does not exist'
    assert os.path.exists(ann_file), ann_file + ' does not exist'

    # img = mmcv.imread(img_file, channel_order='rgb')
    # plt.imshow(img)
    # plt.show()
    ann = mmcv.imread(ann_file, flag='unchanged', backend='pillow')
    for val in np.unique(ann):
        cnt[val] += np.sum(ann == val)

with open(ANN_FILES2, "r") as f:
    content = f.read()
files = content.split("\n")
files.remove('')
for file in files:
    img_file = os.path.join(IMG_PATH, file + '.jpg')
    ann_file = os.path.join(ANN_DIR2, file + '.png')
    assert os.path.exists(img_file), img_file + ' does not exist'
    assert os.path.exists(ann_file), ann_file + ' does not exist'

    # img = mmcv.imread(img_file, channel_order='rgb')
    # plt.imshow(img)
    # plt.show()
    ann = mmcv.imread(ann_file, flag='unchanged', backend='pillow')
    for val in np.unique(ann):
        cnt[val] += np.sum(ann == val)

n_samples = 0
for val in values[:-1]:
    n_samples += cnt[val]

weights = np.empty(len(values[:-1]))
for val in values[:-1]:
    weights[val] = n_samples / (len(values[:-1]) * cnt[val])
np.save('/data/dataset/VOCdevkit/VOC_seg_weights.npy', weights)

