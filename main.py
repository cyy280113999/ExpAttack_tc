from functools import partial
from itertools import product
from utils import *
from WAMethods import *
from evalAttacks import evaluate_attack_success_rate

pj = os.path.join

# models that prepared to generate
# hint:[model, layer] source model, layer for feature-level attack
available_source_models = {
    'vgg16_m3': ['vgg16', (('features', 16),)],
    'vgg19_c3': ['vgg19', (('features', 17),)],
    'vgg19_m3': ['vgg19', (('features', 18),)],
    'res50_b2u8': ['res50', (('layer2', 8, 'relu2'),)],
}
# =========================generate settings====================================
# adv-sample settings
eps = 16  # at 255 level
default_params = {'eps': eps / 255,
                  'steps': 10,
                  'alpha': eps / 255 / 10,
                  'activation_class': LayerActivation,
                  'updater_class': L1_M11_S,
                  }

# ==========================evaluate settings============================
# models to evaluate
model_names = [
    "vgg16",
    "vgg19",
    "res50",
    "res152",
    "inc1",
    "inc3",
    "inc4",
    "dense121",
    "convnext",
    "vit",
    "deit",
    "swin",
]

available_expriments = {
    'FGSM': {
        'steps': 1,
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_S,
    },
    'BIM': {
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_S,
    },
    'MIM': {
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_M11_S,
    },
    'FIA': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
    },
    'FIA_NM_A10': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 1.0,
        'updater_class': M11_N,
    },
    'FIA_NM_A15': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 1.5,
        'updater_class': M11_N,
    },
    'FIA_NM_A20': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 2.0,
        'updater_class': M11_N,
    },
    'FIA_NM_A25': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 2.5,
        'updater_class': M11_N,
    },
    'FIA_NM_A30': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 3.0,
        'updater_class': M11_N,
    },
    'FIA_NM_A35': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 3.5,
        'updater_class': M11_N,
    },
    'FIA_NM_A40': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 4.0,
        'updater_class': M11_N,
    },
    'FIA_NM_A45': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 4.5,
        'updater_class': M11_N,
    },
    'FIA_NM_A50': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 5.0,
        'updater_class': M11_N,
    },
    'FIA_NM_A55': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 5.5,
        'updater_class': M11_N,
    },
    'FIA_NM_A60': {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        'alpha': eps / 255 / 10 * 6.0,
        'updater_class': M11_N,
    },

    # 'FIA_DIM_MIM_TIM_PIM': {
    #     'DIM_caller': DIM(),
    #     'weight_class': partial(FIAWeight, mode='CE'),
    #     'activation_class':LayerActivation,
    #     'update_class': lambda alpha:
    #     SequenceClass([ListOperation([Operation(),TIM()]),
    #                    ListOperation([Operation(),Momentum()]),
    #                    PIM_Updater(alpha)]),
    # },
    'NAA': {
        'weight_class': partial(NAAWeight, mode='CE'),
    },
    # 'FMAA_P69M1110': {
    #     'weight_class': partial(FMAAWeight, prob1=0.6, prob2=0.9, mode='CE', decay=1.1, keep=1),
    #     'updater_class': L1_M11_S,
    # },
    # 'FMAA11_vgg_m3': (WA, {
    #     'weight_class': partial(FMAAWeight, prob1=0.6, prob2=0.9, mode='CE', decay=1, keep=1),
    #     'updater_class': L1_M11_S,
    # }),
    # 'FMAA91_vgg_m3': (WA, {
    #     'weight_class': partial(FMAAWeight, prob1=0.6, prob2=0.9, mode='CE',decay=0.9,keep=0.1),
    #     'updater_class': L1_M11_S,
    # }),

    # 'NAACE_vgg_m3': (WA, {
    #     'weight_class': partial(NAAWeight, mode='CE'),
    # }),
    'LID_TT': {
        'weight_class': partial(LIDIGWeight, BP="normal", LIN=1, DEFAULT_STEP=11, GIP=0.3, ),
    },
    'LID_II': {
        'weight_class': partial(LIDIGWeight, BP="sig", LIN=0, DEFAULT_STEP=11, GIP=0.3,),
    },
}


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


def generate(method, dataset, save_dir, noise_mode=False):
    tqdm.write(f'generating {save_dir}')
    for i in tqdm(range(len(dataset))):
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

def generate_and_evaluate(select_expriments=None, select_source_models=None):
    """*****************************************************************************************************************
    ********************************              main              ****************************************************
    *****************************************************************************************************************"""
    regenerating = True
    evaluating = True
    # generate from
    select_source_models = ['res50_b2u8', ]
    # run what experiments
    select_expriments = [
        # 'FIA_NM_A10',
        # 'FIA_NM_A15',
        # 'FIA_NM_A20',
        # 'FIA_NM_A25',
        # 'FIA_NM_A30',
        # 'FIA_NM_A35',
        # 'FIA_NM_A40',
        # 'FIA_NM_A45',
        # 'FIA_NM_A50',
        # 'FIA_NM_A55',
        # 'FIA_NM_A60',
    ]
    select_log_file = None  # auto logging with settings
    ds = FIA_Dataset()

    # =============================== run =======================================
    for model_key, method_name in product(select_source_models, select_expriments, ):
        # ==========generating==========
        # save to
        adv_root = f'adv_{model_key}_e{eps}'
        adv_dir = pj(adv_root, method_name)
        if regenerating:
            model_name, layer_names = available_source_models[model_key]
            model = get_model(model_name)
            layers = auto_hook(model, layer_names)
            params = default_params.copy()
            params_appendix = available_expriments[method_name]
            params.update(params_appendix)
            method = WA(model=model, layers=layers, **params)
            generate(method, ds, adv_dir)
            clearHooks(model)
            del model
        # =========evaluating=========
        # log to
        if select_log_file is not None:
            log_file = select_log_file
        else:
            log_file = f'{adv_root}.csv'
        if evaluating:
            evaluate_attack_success_rate(model_names, adv_dir, log_file)


if __name__ == '__main__':
    generate_and_evaluate()
