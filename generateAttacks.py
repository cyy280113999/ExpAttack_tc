import os

import torch
from PIL import Image
import torchattacks as ta
from NAA import NAA
from utils import *
from MIMU import MIMU
from FIA import FIA
pj = os.path.join


class FIA_Dataset:
    """
    copy the corresponding directory ./dataset.
    copy the labels.txt to ./dataset

    ds_dir: ./dataset
    """

    def __init__(self, ds_dir='./dataset'):
        with open(pj(ds_dir, 'labels.txt')) as f:
            self.labels = [int(s) - 1 for s in f.read().split('\n')[:-1]]  # change the start point from 1 to 0
        self.ds_dir = ds_dir
        self.image_names = os.listdir(pj(ds_dir, 'images'))
        self.image_names.sort(key=lambda s: int(s[:-4]))

    def __getitem__(self, i):
        img = pilOpen(pj(self.ds_dir, 'images', self.image_names[i]))
        img = toTensorS224(img)
        lb = self.labels[i]
        return self.image_names[i], img, lb

    def __len__(self):
        return 1000


def generate(method, dataset, save_dir):
    for i in tqdm(range(len(dataset))):
        file_name, x, y = dataset[i]
        x = toStd(x).unsqueeze(0).cuda()
        y = torch.LongTensor([y]).cuda()
        adv = method(x, y)
        # adv = method(*preprocess(x, y))
        adv = toPlot(invStd(adv.cpu()))
        save_image(adv, file_name, save_dir)


def showTop10(model, x):
    y = model(x)
    indices = y.argsort(dim=1, descending=True)[0, :10].cpu().numpy()
    for i in indices:
        print(i, y[0, i].item())


def preprocess(x, y):
    x = toStd(x)
    x = x.cuda()
    return x, y


def save_image(image, name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Image.fromarray((image * 255).astype('uint8'))
    img.save(pj(output_dir, name))


def ta_wrp(ta_cls, **kwargs):
    def class_wrapper(model):
        atk = ta_cls(model, **kwargs)  # predefine
        atk.set_normalization_used(ImgntMean, ImgntStd)

        def method(x, y):
            return atk(x, y)

        return method

    return class_wrapper


exps = {
    'FGSM_vgg': (ta.FGSM, 'vgg16', {'eps': 16 / 255}),

    # 'PGD': ta_wrp(ta.PGD, **{
    #     'eps': 16 / 255,
    #     'alpha': 1.6 / 255,
    #     'steps': 10,
    #     'random_start': False,
    # }),
    # 'MIM': ta_wrp(ta.MIFGSM, **{
    #     'eps': 16 / 255,
    #     'alpha': 1.6 / 255,
    #     'steps': 10,
    # }),
    # 'MIMU': ta_wrp(MIMU, **{
    #     'eps': 16 / 255,
    #     'alpha': 1.6 / 255,
    #     'steps': 10,
    # }),
    # 'FIA': ta_wrp(FIA, **{
    #     'layer_name': (('features',16),),
    # }),
    'FIA_vgg_c2': (FIA, 'vgg16', {'layer_name': (('features', 8),)}),
    'FIA_vgg_m2': (FIA, 'vgg16', {'layer_name': (('features', 9),)}),
    'FIA_vgg_c3': (FIA, 'vgg16', {'layer_name': (('features', 15),)}),  # conv33
    'FIA_vgg_m3': (FIA, 'vgg16', {'layer_name': (('features', 16),)}),  # maxpool3
    'FIA_vgg_c4': (FIA, 'vgg16', {'layer_name': (('features', 22),)}),
    'FIA_vgg_m4': (FIA, 'vgg16', {'layer_name': (('features', 23),)}),
    'NAA_vgg_c2': (NAA, 'vgg16', {'layer_name': (('features', 8),)}),
    'NAA_vgg_m2': (NAA, 'vgg16', {'layer_name': (('features', 9),)}),
    'NAA_vgg_c3': (NAA, 'vgg16', {'layer_name': (('features', 15),)}),  # conv33
    'NAA_vgg_m3': (NAA, 'vgg16', {'layer_name': (('features', 16),)}),  # maxpool3
    'NAA_vgg_c4': (NAA, 'vgg16', {'layer_name': (('features', 22),)}),
    'NAA_vgg_m4': (NAA, 'vgg16', {'layer_name': (('features', 23),)}),
}

if __name__ == '__main__':
    method_name='FIA_vgg_m4'
    method_class,model_name,params=exps[method_name]
    ds = FIA_Dataset()
    model = get_model(model_name)
    method = method_class(model, **params)
    method.set_normalization_used(ImgntMean, ImgntStd)
    save_dir = f'./adv/tc_{method_name}'
    generate(method, ds, save_dir)
