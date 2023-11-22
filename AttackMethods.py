import torch
import numpy as np
from utils import *
from methods.LIDDecomposer import LIDDecomposer


def Raw_Activation(self, model, layers, x, y=None):
    as_ = []
    model(toStd(x))
    for l in layers:
        as_.append(l.activation)
    return as_


def Diff_Activation(self, model, layers, x, y=None):
    as0 = []
    as1 = []
    model(toStd(torch.zeros_like(x)))  # std0. black
    for l in layers:
        as0.append(l.activation)
    model(toStd(x))
    for l in layers:
        as1.append(l.activation)
    as_ = [a1 - a0 for a1, a0 in zip(as1, as0)]
    return as_


# def FGSM_SwitchGradient(model, layers, x, y):
#     g = torch.nn.functional.one_hot(y, 1000)
#     return [g]

class FIA_AggragatedGradient:
    def __init__(self, post_softmax=False):
        self.post_softmax = post_softmax

    def __call__(self, model, layers, x, y):
        gs = [list() for _ in range(len(layers))]
        if not self.post_softmax:
            pred_fun = lambda x: model(toStd(x))[:, y]
        else:
            pred_fun = lambda x: model(toStd(x)).softmax(1)[:, y]
        for j in range(30):
            mask = torch.tensor(np.random.binomial(1, 0.9, size=x.shape)).cuda()  # binomial
            ran = x * mask
            ran = ran.detach().requires_grad_()
            pred_fun(ran).sum().backward()
            for i, layer in enumerate(layers):
                gs[i].append(layer.gradient)
        for i, g in enumerate(gs):
            gs[i] = torch.mean(torch.vstack(g), dim=[0], keepdim=True)
        return gs


class NAA_IntegratedGradient:
    def __init__(self, post_softmax=True):
        self.post_softmax = post_softmax

    def __call__(self, model, layers, x, y):
        gs = [list() for _ in range(len(layers))]
        if not self.post_softmax:
            pred_fun = lambda x: model(toStd(x))[:, y]
        else:
            pred_fun = lambda x: model(toStd(x)).softmax(1)[:, y]
        for j in range(30):
            ratio = (i + 1) / 10
            ran = ratio * x
            ran = ran.detach().requires_grad_()
            pred_fun(ran).sum().backward()
            for i, layer in enumerate(layers):
                gs[i].append(layer.gradient)
        for i, g in enumerate(gs):
            gs[i] = torch.mean(torch.vstack(g), dim=[0], keepdim=True)
        return gs


class LID_Gradient:
    def __call__(self, model, layers, x, y):
        method = LIDDecomposer(model, LIN=0, bp='sig')
        x = x.detach().requires_grad_()
        method(toStd(x), y)
        ag = []
        for i, layer in enumerate(layers):
            ag.append(layer.g)
        return ag


def norm(g, level=1):
    if level == 0:
        pass
    elif level == 1:
        sum_abs = g.abs().sum([1, 2, 3], True)
        assert not torch.any(sum_abs == 0)
        g = g / sum_abs
    elif level == 2:
        sum_sqr = g.square().sum([1, 2, 3], True)
        assert not torch.any(sum_sqr == 0)
        g = g / sum_sqr.sqrt()
    elif level == 'inf':
        M = g.abs().max([1, 2, 3], True)
        g = g / M
    return g


class Momentum:
    def __init__(self, decay=0.9, keep=0.1):
        self.decay = decay
        self.keep = keep
        self.accu_gradient = None

    def __call__(self, gradient):
        if self.accu_gradient is None:
            self.accu_gradient = torch.zeros_like(gradient)
        self.accu_gradient = self.decay * self.accu_gradient + self.keep * gradient
        return self.accu_gradient

    def zeros(self):
        self.accu_gradient = None


class GD_Updater:
    def __call__(self, x, g, alpha):
        return x - alpha * g


class FGSM_Updater:
    def __call__(self, x, g, alpha):
        return x - alpha * g.sign()


class FGNM_Updater:
    def __call__(self, x, g, alpha):
        k = g.sign().abs().sum([1, 2, 3], True).sqrt()
        return x - alpha * k * norm(g, 2)


class GD:
    def __init__(self):
        self.updater = GD_Updater()

    def __call__(self, x, g, alpha):
        x = self.updater(x, g, alpha)
        return x


# sign: wrong direction
# L2: wrong step
class L2_M11_S:
    def __init__(self):
        self.momentum = Momentum(1, 1)
        self.updater = FGSM_Updater()

    def __call__(self, x, g, alpha):
        g = norm(g, 2)
        m = self.momentum(g)
        x = self.updater(x, m, alpha)
        return x

    def zeros(self):
        self.momentum.zeros()


class M11_S:
    def __init__(self):
        self.momentum = Momentum(1, 1)
        self.momentum.zeros()
        self.updater = FGSM_Updater()

    def __call__(self, x, g, alpha):
        m = self.momentum(g)
        x = self.updater(x, m, alpha)
        return x

    def zeros(self):
        self.momentum.zeros()


class L2_M11_N:
    def __init__(self):
        self.momentum = Momentum(1, 1)
        self.updater = FGNM_Updater()

    def __call__(self, x, g, alpha):
        g = norm(g, 2)
        m = self.momentum(g)
        x = self.updater(x, m, alpha)
        return x

    def zeros(self):
        self.momentum.zeros()


class M11_N:
    def __init__(self):
        self.momentum = Momentum(1, 1)
        self.updater = FGNM_Updater()

    def __call__(self, x, g, alpha):
        m = self.momentum(g)
        x = self.updater(x, m, alpha)
        return x

    def zeros(self):
        self.momentum.zeros()


class M11_L2:
    def __init__(self):
        self.momentum = Momentum(1, 1)
        self.updater = GD_Updater()

    def __call__(self, x, g, alpha):
        m = self.momentum(g)
        m = norm(m, 2)
        x = self.updater(x, m, alpha)
        return x

    def zeros(self):
        self.momentum.zeros()


class WA:
    """
    weighted-activations attack

    loop n times:
        get weights
        get activation
        loss = weighted activations
        get grad of loss wrt input
        update input by subtracting grad(momentum)
    """

    def __init__(self, model,
                 eps=8 / 255,  # bound
                 steps=10, alpha=None,  # iterate
                 layer_names=None,  # activations
                 weight_function=None,  # weights
                 update_weight=False,
                 activation_function=None,
                 updater_class=None,  # momentum
                 **kwargs):
        super().__init__("WA", model)
        self.supported_mode = ["default", "targeted"]
        self.device = 'cuda'
        self.model = model
        self.eps = eps  # bound
        self.steps = steps  # iterate
        self.alpha = alpha  # given alpha
        if alpha is None:  # auto alpha
            self.alpha = self.eps / self.steps
        self.layers = auto_hook(model, layer_names)  # activations
        self.weight_function = weight_function  # weights
        self.update_weight = update_weight
        self.activation_function = activation_function
        self.updater = updater_class()

    def __call__(self, images, labels):
        images = images.clone().detach().to(self.device)  # [0,1]. un-normalized
        adv_images = images.clone().detach()
        labels = labels.clone().detach().to(self.device)
        self.updater.zeros()
        for i in range(self.steps):
            if i == 0 or self.update_weight:
                gs = self.weight_function(self.model, self.layers, adv_images, labels)
            as_ = self.activation_function(self.model, self.layers, adv_images, labels)
            loss = sum((a * g).sum() for a, g in zip(as_, gs))
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = self.updater(adv_images, grad)
            adv_images = torch.clamp(adv_images, min=images - self.eps, max=images + self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images.detach()

    def __del__(self):
        clearHooks(self.model)
