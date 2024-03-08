from functools import partial

from utils import *
from WAMethods import *

pj = os.path.join


class FIA_Dataset:
    """
    copy the corresponding directory ./dataset.
    copy the labels.txt to ./dataset

    ds_dir: ./dataset
    """

    def __init__(self, ds_dir='./dataset'):
        self.ds_dir = ds_dir
        self.image_names = os.listdir(pj(ds_dir, 'images'))
        self.image_names.sort(key=lambda s: int(s[:-4]))  # images inorder
        with open(pj(ds_dir, 'labels.txt')) as f:
            self.labels = [int(s) - 1 for s in f.read().split('\n')[:-1]]  # change the start point from 1 to 0

    def __getitem__(self, i):
        img = pilOpen(pj(self.ds_dir, 'images', self.image_names[i]))
        img = toTensorS224(img)
        lb = self.labels[i]
        return self.image_names[i], img, lb

    def __len__(self):
        return 1000


# generate all adversal samples of the specific method
def generate(method, dataset, save_dir, noise_mode=False):
    for i in tqdm(range(len(dataset)), desc=save_dir):
        file_name, x, y = dataset[i]
        x = x.unsqueeze(0).cuda()
        y = torch.LongTensor([y]).cuda()
        noise = method(x, y)
        adv = (x + noise).clip(min=0, max=1).cpu()
        save_image(adv, pj(save_dir, file_name))


# show top10 predictions to check the success of attacking
def showTop10(model, x):
    y = model(x)
    indices = y.argsort(dim=1, descending=True)[0, :10].cpu().numpy()
    for i in indices:
        print(i, y[0, i].item())


def save_image(image, name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Image.fromarray((image * 255).astype('uint8'))
    img.save(pj(output_dir, name))

# def eval_all():
#     ds = FIA_Dataset()
#     model_name = 'vgg16'
#     default_params = {'eps': 8 / 255,
#                       'layer_names': (('features', 16),),
#                       'activation_class': Diff_Activation,
#                       'updater_class': M11_S,
#                       }
#     # all the experiments
#     exps = {
#         'FIACE_vgg_m3': {
#             'weight_class': partial(FIAWeight, mode='CE'),
#         },
#         # -- variant
#         # 'FIACEUW_vgg_m3': (WA,{
#         #     'weight_function': FIA_AggragatedGradient(mode='CE'),
#         #     'update_weight': True,
#         # }),
#
#         # NAA
#         # 'NAACE_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE'),
#         # }),
#
#         # 'NAACEUW_vgg_m3': (WA, {
#         #     'weight_function': NAA_IntegratedGradient(mode='CE'),
#         #     'update_weight': True,
#         # }),
#         # 'NAAC0_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=0),
#         # }),
#         # 'NAAC5_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=5),
#         # }),
#         # 'NAAC10_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=10),
#         # }),
#         # 'NAAC15_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=15),
#         # }),
#         # 'NAAC20_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=20),
#         # }),
#         # 'NAAC25_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=25),
#         # }),
#         # 'NAAC30_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAWeight, mode='CE', weight_center=30),
#         # }),
#         # LID
#         # 'LIDTT_vgg_m3': (WA, {
#         #     'weight_function': LID_Gradient(BP='normal', LIN=1),
#         # }),
#         # 'LIDIIUW_vgg_m3': (WA, {
#         #     'weight_function': LID_Gradient(BP='sig',LIN=0),
#         #     'update_weight': True,
#         # }),
#         # 'LIDITUW_vgg_m3': (WA, {
#         #     'weight_function': LID_Gradient(BP='sig', LIN=1),
#         #     'update_weight': True,
#         # }),
#         # 'LIDTIUW_vgg_m3': (WA, {
#         #     'weight_function': LID_Gradient(BP='st', LIN=0),
#         #     'update_weight': True,
#         # }),
#         # 'LIDIIUW_S20_vgg_m3': (WA, {
#         #     'weight_function': LID_Gradient(BP='sig', LIN=0,DEFAULT_STEP=21),
#         #     'update_weight': True,
#         # }),
#         # 'LIDIIUW_S30_vgg_m3': (WA,{
#         #     'weight_function': LID_Gradient(BP='sig', LIN=0,DEFAULT_STEP=31),
#         #     'update_weight': True,
#         # }),
#         # 'LIDII_FGSM_vgg_m3': (WA,{
#         #     'steps':1,  # FGSM
#         #     'weight_function': LID_Gradient(BP='sig', LIN=0),
#         # }),
#         # 'LIDII_GIPS30_vgg_m3': (WA,{
#         #     'weight_function': LID_Gradient(BP='sig', LIN=0, GIP=0.3, DEFAULT_STEP=31),
#         # }),
#         # 'LIDIG_GIPS30_vgg_m3': (WA, {
#         #     'weight_function': LIDIG_Gradient(BP='sig', LIN=0, GIP=0.3, DEFAULT_STEP=31),
#         # }),
#         # 'MM55_vgg_m3': (WA, {
#         #     'weight_class': partial(MidMomentumWeight, mode='CE', decay=0.5, keep=0.5),
#         # }),
#         # 'MM91_vgg_m3': (WA, {
#         #     'weight_class': partial(MidMomentumWeight, mode='CE', decay=0.9, keep=0.1),
#         # }),
#         # 'MM10_vgg_m3': (WA, {
#         #     'weight_class': partial(MidMomentumWeight, mode='CE', decay=1, keep=0),
#         # }),
#         # 'FIAM55_vgg_m3': (WA, {
#         #     'weight_class': partial(FIAMWeight, mode='CE', decay=0.5, keep=0.5),
#         # }),
#         # 'FIAM91_vgg_m3': (WA, {
#         #     'weight_class': partial(FIAMWeight, mode='CE', decay=0.9, keep=0.1),
#         # }),
#         # 'FIAM10_vgg_m3': (WA, {
#         #     'weight_class': partial(FIAMWeight, mode='CE', decay=1, keep=0),
#         # }),
#         # 'NAAM55_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAMWeight, mode='CE', decay=0.5, keep=0.5),
#         # }),
#         # 'NAAM91_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAMWeight, mode='CE', decay=0.9, keep=0.1),
#         # }),
#         # 'NAAM10_vgg_m3': (WA, {
#         #     'weight_class': partial(NAAMWeight, mode='CE', decay=1, keep=0),
#         # }),
#         # 'LIDIGF0_vgg_m3': (WA, {
#         #     'weight_class': partial(LIDIGWeight,BP='sig', LIN=0, FracLevel=0),
#         # }),
#         # 'LIDIGF1_vgg_m3': (WA, {
#         #     'weight_class': partial(LIDIGWeight,BP='sig', LIN=0, FracLevel=1),
#         # }),
#         # 'LIDIGF2_vgg_m3': (WA, {
#         #     'weight_class': partial(LIDIGWeight,BP='sig', LIN=0, FracLevel=2),
#         # }),
#         # 'LIDIGF3_vgg_m3': (WA, {
#         #     'weight_class': partial(LIDIGWeight,BP='sig', LIN=0, FracLevel=3),
#         # }),
#         # 'LIDIGF4_vgg_m3': (WA, {
#         #     'weight_class': partial(LIDIGWeight,BP='sig', LIN=0, FracLevel=4),
#         # }),
#     }
#     print(exps.keys())
#     for method_name in exps:
#         params = default_params.copy()
#         params_appendix = exps[method_name]
#         model = get_model(model_name)
#         params.update(params_appendix)
#         method = WA(model, **params)
#         save_dir = f'adv_e8/{method_name}'
#         generate(method, ds, save_dir)


# if __name__ == '__main__':
#     eval_all()
