from generateAttacks import *

model_name = 'vgg16'
default_params = {'eps': 8 / 255,
                  'layer_names': (('features', 16),),
                  'activation_class': Diff_Activation,
                  'updater_class': L1_M11_S,
                  }
exps = {
    'FGSM_vgg_m3': (WA, {
        'steps': 1,
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_S,
    }),
    'BIM_vgg_m3': (WA, {
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_S,
    }),
    'MIM_vgg_m3': (WA, {
        'weight_class': partial(LayerWeight, mode='CE'),
        'updater_class': L1_M11_S,
    }),
    'FIA_vgg_m3': (WA, {
        'weight_class': partial(FIAWeight, mode='CE'),  # use negative cross-entropy loss
    }),

}