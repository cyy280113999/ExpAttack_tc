import csv
import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from utils import *

pj = os.path.join


class EvalDataset:
    def __init__(self, ds_dir='./dataset'):
        self.ds_dir = ds_dir
        self.image_names = os.listdir(ds_dir)
        self.image_names.sort(key=lambda s: int(s[:-4]))  # images inorder

    def __getitem__(self, i):
        img = pilOpen(pj(self.ds_dir, self.image_names[i]))
        img = toTensorS224(img)
        return img

    def __len__(self):
        return 1000


def evaluate_attack_success_rate_on_model(clean_images, adv_images, model):
    total_samples = len(clean_images)
    successful_attacks = 0
    clean_loader = TD.DataLoader(clean_images,batch_size=16,shuffle=False)
    adv_loader = TD.DataLoader(adv_images,batch_size=16,shuffle=False)
    for clean_batch, adv_batch in tqdm(zip(clean_loader, adv_loader)):
        clean_batch = toStd(clean_batch).cuda()
        adv_batch = toStd(adv_batch).cuda()

        clean_predictions = model(clean_batch)
        adv_predictions = model(adv_batch)

        clean_labels = torch.argmax(clean_predictions, dim=1).cpu().numpy()
        adv_labels = torch.argmax(adv_predictions, dim=1).cpu().numpy()

        successful_attacks += sum(clean_label != adv_label for clean_label, adv_label in zip(clean_labels, adv_labels))

    attack_success_rate = successful_attacks / total_samples
    return attack_success_rate


def evaluate_attack_success_rate(model_names, adv_dir, csv_dir):
    tqdm.write(adv_dir)
    clean_data_path = 'dataset/images/'  # fix
    acc = []
    clean_images = EvalDataset(clean_data_path)
    adv_images = EvalDataset(adv_dir)
    saver = ResultSaver(model_names, adv_dir, csv_dir)
    for model_name in model_names:
        tqdm.write(model_name)
        model = get_model(model_name)
        success_rate = evaluate_attack_success_rate_on_model(clean_images, adv_images, model)
        acc.append(f'{success_rate:.1%}')
    saver.save(acc)
    tqdm.write(','.join(acc))


class ResultSaver:
    def __init__(self, model_names, adv_dir, csv_dir):
        self.adv_dir = adv_dir
        self.csv_dir = csv_dir
        # Check if the CSV file already exists
        if not os.path.isfile(csv_dir):
            # Create the CSV file and write the header
            with open(csv_dir, 'w') as file:
                file.write("name," + ','.join(model_names)+'\n')

    def save(self, success_rate):
        # Append the data to the CSV file
        with open(self.csv_dir, 'a') as file:
            file.write(f"{self.adv_dir}," + ','.join(success_rate) + '\n')

# # 设置 HTTP 代理
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:47890'
# # 设置 HTTPS 代理
# os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:47890'
def main():
    # adv_dirs = ['dataset/images',]  # clean input, non-adv.
    # adv_root='adv_e16'
    # adv_dirs = [f'{adv_root}/{d}' for d in os.listdir(adv_root)]  # scan
    # adv_dirs =['LIDTIUW_vgg_m3', 'LIDIIUW_S20_vgg_m3', 'LIDIIUW_S30_vgg_m3', 'LIDII_FGSM_vgg_m3']

    adv_root='adv_e8'
    adv_dirs =['NAAC0_vgg_m3', 'NAAC5_vgg_m3', 'NAAC10_vgg_m3', 'NAAC15_vgg_m3', 'NAAC20_vgg_m3', 'NAAC25_vgg_m3', 'NAAC30_vgg_m3']
    adv_dirs = [pj(adv_root,x) for x in adv_dirs]
    csv_dir = 'log_e8.csv'

    model_names = [
        "vgg16",
        "vgg19",
        "resnet50",
        "resnet152",
        "googlenet",
        "inception3",
        "inception4",
        "densenet121",
        "convnext",
        "vit",
        "deit",
        "swin",
    ]
    for adv_dir in adv_dirs:
        evaluate_attack_success_rate(model_names, adv_dir, csv_dir)


if __name__ == '__main__':
    main()
