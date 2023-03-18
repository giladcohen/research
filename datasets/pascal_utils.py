from typing import Dict
import matplotlib.pyplot as plt


def set_scaled_img(data: Dict):
    x = data['x']
    minn = x.min()
    maxx = x.max()
    scaled_x = (x - minn) / (maxx - minn)
    data['scaled_x'] = scaled_x
    data['meta'].update({'minn': minn.item(), 'maxx': maxx.item()})

def unscale(x, minn, maxx):
    x *= (maxx - minn)
    x += minn
    return x

def parse_data(data):
    data['x'] = data['img'][0]
    data['meta'] = data['img_metas'][0]._data[0][0]

def verify_data(data):
    imgs = data['img']
    img_metas = [im._data for im in data['img_metas']][0]
    for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
        if not isinstance(var, list):
            raise TypeError(f'{name} must be a list, but got f{type(var)}')
    num_augs = len(imgs)
    if num_augs != len(img_metas):
        raise ValueError(f'num of augmentations ({len(imgs)}) != ' + f'num of image meta ({len(img_metas)})')
    # all images in the same aug batch all of the same ori_shape and pad shape
    for img_meta in img_metas:
        ori_shapes = [_['ori_shape'] for _ in img_meta]
        assert all(shape == ori_shapes[0] for shape in ori_shapes)
        img_shapes = [_['img_shape'] for _ in img_meta]
        assert all(shape == img_shapes[0] for shape in img_shapes)
        pad_shapes = [_['pad_shape'] for _ in img_meta]
        assert all(shape == pad_shapes[0] for shape in pad_shapes)

    assert num_augs == 1, 'currently not supporting TTAs'

def show_rgb_img(img):
    x = img.copy()
    x[:, :, 0] = img[:, :, 2]
    x[:, :, 2] = img[:, :, 0]
    plt.imshow(x)
    plt.show()
