from generateAttacks import *
from evalAttacks import *

# models that prepared to generate
# hint:[model, layer]
source_models = {
    'vgg16_m3': ['vgg16',(('features', 16),)],
    'vgg19_c3': ['vgg19',(('features', 17),)],
    'vgg19_m3': ['vgg19',(('features', 18),)],
}


def eval_and_test(model_and_layer='vgg19_m3'):
    model_and_layer = 'vgg19_m3'
    # =============================generating====================================
    ds = FIA_Dataset()
    # source model. layer for feature-level attack
    model_name, layer_names = source_models[model_and_layer]
    # adv-sample settings
    eps = 16  # at 255 level
    default_params = {'eps': eps/255,
                      'steps': 10,
                      'alpha': eps/255/10,
                      'activation_class': LayerActivation,
                      'updater_class': L1_M11_S,
                      }
    # fix attacker
    Attacker = WA
    # all the experiments
    exps = {
        # 'FIA': {
        #     'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
        # },
        # 'NAA': {
        #     'weight_class': partial(NAAWeight, mode='CE'),  # use negative cross-entropy loss
        # },
        'FMAA_P69M1110': {
            'weight_class': partial(FMAAWeight, prob1=0.6, prob2=0.9, mode='CE', decay=1.1, keep=1),
            'updater_class': L1_M11_S,
        },
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
        # 'LIGF0R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=0, FracLevel=0, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LIGF1R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=0, FracLevel=1, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LIGF2R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=0, FracLevel=2, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LIGF3R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=0, FracLevel=3, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LIGF4R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=0, FracLevel=4, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LINF0R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=1, FracLevel=0, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LINF1R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=1, FracLevel=1, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LINF2R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=1, FracLevel=2, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LINF3R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=1, FracLevel=3, DEFAULT_STEP=31, GIP=0.3),
        # }),
        # 'LINF4R_vgg_m3': (WA, {
        #     'weight_class': partial(LIDIGWeight, BP='sig', LIN=1, FracLevel=4, DEFAULT_STEP=31, GIP=0.3),
        # }),
    }
    adv_root = f'adv_{model_and_layer}_e{eps}'
    csv_file = f'{adv_root}.csv'
    # ======================================eval===========================================
    model_names = [
        "vgg16",
        "vgg19",
        "res50",
        "res152",
        "inc1",
        "inc3",
        "inc4",
        # 'incres2'
        "dense121",
        "convnext",
        "vit",
        "deit",
        "swin",
    ]
    for method_name in exps:
        model = get_model(model_name)
        layers = auto_hook(model, layer_names)
        params = default_params.copy()
        params_appendix = exps[method_name]
        params.update(params_appendix)
        method = Attacker(model=model, layers=layers, **params)
        adv_dir = pj(adv_root, method_name)
        generate(method, ds, adv_dir)
        clearHooks(model)
        del model
        evaluate_attack_success_rate(model_names, adv_dir, csv_file)


if __name__ == '__main__':
    eval_and_test()
