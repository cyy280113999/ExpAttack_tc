import torch
import numpy as np
from utils import *
from methods.LIDDecomposer import LIDDecomposer
from methods.LIDIG import LIDIG

class _Operation:
    def __call__(self, x):
        return x
    def zeros(self):
        pass

class _Wrapper(_Operation):
    def __call__(self, x):
        return self.obj(x)
    def zeros(self):
        if hasattr(self.obj,'zeros'):
            self.obj.zeros()

class ListOperation(_Operation):
    def __init__(self, ops):
        self.ops=ops
    def zeros(self):
        for f in self.ops:
            if hasattr(f,'zeros'):
                f.zeros()
    def __call__(self, xs):
        ys=[]
        for i, x in enumerate(xs):
            f = self.ops[i % len(self.ops)]
            ys.append(f(x))
        return ys

class Fixer(_Operation):
    def __init__(self, op):
        self.mem = None
        self.fix = False
        self.op=op
    def zeros(self):
        self.mem = None
        self.fix = False
        if hasattr(self.op, 'zeros'):
            self.op.zeros()
    def __call__(self, x):
        if not self.fix:
            self.mem = self.op(x) # keep first input
            self.fix=True
        return self.mem


class Momentum(_Operation):
    def __init__(self, decay=0.9, keep=1):
        self.decay = decay
        self.keep = keep
        self.mem = None

    def zeros(self):
        self.mem = None

    def __call__(self, gradient):
        if self.mem is None:
            self.mem = torch.zeros_like(gradient)
        self.mem = self.decay * self.mem + self.keep * gradient
        return self.mem


class LayerWiseClass(ListOperation):
    def __init__(self, single):
        super().__init__(single)
        self.single=single
    def zeros(self):
        if hasattr(self.single,'zeros'):
            self.single.zeros()
    def __call__(self, xs):
        ys=[]
        for x in xs:
            ys.append(self.single(x))
        return ys

class SequenceClass(_Operation):
    def __init__(self, seq):
        self.seq=seq

    def zeros(self):
        for item in self.seq:
            if hasattr(item,'zeros'):
                item.zeros()

    def __call__(self, x):
        for item in self.seq:
            x = item(x)
        return x

# activation implement
class LayerActivation(_Operation):
    def __init__(self, model, layers):
        self.model=model
        self.layers=layers
    def __call__(self, x):
        la = []
        self.model(toStd(x))
        for l in self.layers:
            la.append(l.activation)
        return la

class FixActivation(_Wrapper):
    def __init__(self, model, layers):
        super().__init__()
        self.obj=SequenceClass([
            Fixer(LayerActivation(model, layers)),
        ])


class Diff_Activation(_Operation):
    def __init__(self, model, layers):
        self.model=model
        self.layers=layers
    def __call__(self, x):
        la0 = []
        self.model(toStd(torch.zeros_like(x)))  # std0. black
        for l in self.layers:
            la0.append(l.activation)
        la1 = []
        self.model(toStd(x))
        for l in self.layers:
            la1.append(l.activation)
        la = [a1 - a0 for a1, a0 in zip(la1, la0)]  # diff
        return la


# def FGSM_SwitchGradient(model, layers, x, y):
#     g = torch.nn.functional.one_hot(y, 1000)
#     return [g]

# weight implement
def attack_loss(model, mode):
    if mode in ['logits', 'Z']:
        return lambda x, y: model(toStd(x))[0, y]
    elif mode in ['prob', 'P']:
        return lambda x, y: model(toStd(x)).softmax(1)[0, y]
    elif mode in ['CE', 'NCE']:
        return lambda x, y: -nf.cross_entropy(model(toStd(x)), y)

class LayerWeight(_Operation):
    def __init__(self, model, layers, mode='CE'):
        self.model=model
        self.layers=layers
        self.mode=mode
    def __call__(self, data):
        x, y = data  # weight requires x and y
        loss_fun = attack_loss(self.model, self.mode)
        loss_fun(x, y).sum().backward()
        lg = []
        for i, layer in enumerate(self.layers):
            lg[i].append(layer.gradient)
        return lg

class MidMomentumWeight(_Wrapper):
    def __init__(self, model, layers, decay=0.9, **kwargs):
        super().__init__()
        self.obj = SequenceClass([
            LayerWeight(model, layers, **kwargs),
            LayerWiseClass(Momentum(decay))
        ])

# class FIA_AggragatedGradient:
#     def __init__(self, model, layers, mode=None, noise='binomial', prob=0.9, num=30):
#         self.model=model
#         self.layers=layers
#         self.mode = mode
#         self.prob = prob
#         self.num = num
#
#     def __call__(self, x, y):
#         lg = [list() for _ in range(len(layers))]
#         loss_fun = attack_loss(model, self.mode)
#         for j in range(self.num):
#             mask = torch.tensor(np.random.binomial(1, self.prob, size=x.shape)).cuda()  # binomial
#             ran = x * mask
#             ran = ran.detach().requires_grad_()
#             loss_fun(ran, y).sum().backward()
#             for i, layer in enumerate(layers):
#                 lg[i].append(layer.gradient)
#         for i, gs in enumerate(lg):
#             lg[i] = torch.mean(torch.vstack(gs), dim=[0], keepdim=True)
#         return lg

class BinRanWeight(_Operation):
    # binomial random average gradient
    def __init__(self, model, layers, mode='CE', prob=0.9, num=30):
        self.model=model
        self.layers=layers
        self.mode = mode
        self.prob = prob
        self.num = num

    def BinRan(self, x):
        mask = torch.tensor(np.random.binomial(1, self.prob, size=x.shape)).cuda()  # binomial
        return x * mask

    def __call__(self, data):
        x,y =data
        lg = [list() for _ in range(len(self.layers))]
        loss_fun = attack_loss(self.model, self.mode)
        for j in range(self.num):
            ran = self.BinRan(x)
            ran = ran.detach().requires_grad_()
            loss_fun(ran, y).sum().backward()
            for i, layer in enumerate(self.layers):
                lg[i].append(layer.gradient)
        for i, gs in enumerate(lg):
            gs = torch.vstack(gs)  # list to tensor
            lg[i] = torch.mean(gs, dim=[0], keepdim=True)
        return lg

# this an example of how to create FIA weight by combining the bin-ran and fixer
class FIAWeight(_Wrapper):
    def __init__(self, model, layers, **kwargs):
        self.obj = SequenceClass([
            Fixer(BinRanWeight(model, layers, **kwargs)),
        ])

class FIAMWeight(_Wrapper):
    def __init__(self, model, layers, decay=0.9, **kwargs):
        self.obj = SequenceClass([
            BinRanWeight(model, layers, **kwargs),
            Momentum(decay)
        ])

# tricks
def gaussian_weight(x, center, sigma):
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def generate_weight_list(center, n, sigma=None):
    if sigma is None:
        sigma = n / 10
    weights = [gaussian_weight(opacity, center, sigma) for opacity in range(n)]
    return weights


# class NAA_IntegratedGradient:
#     def __init__(self, model, layers, mode='P', num=30, weight_center=None):
#         self.model=model
#         self.layers=layers
#         self.mode = mode
#         self.num = num
#         self.weight=None
#         if weight_center is not None:
#             self.weight = torch.tensor(generate_weight_list(weight_center, num))
#
#     def __call__(self, x, y):
#         lg = [list() for _ in range(len(layers))]
#         loss_fun = attack_loss(model, self.mode)
#         for i in range(self.num):
#             ratio = (i + 1) / self.num
#             ran = ratio * x
#             ran = ran.detach().requires_grad_()
#             loss_fun(ran, y).sum().backward()
#             for j, layer in enumerate(layers):
#                 lg[j].append(layer.gradient)
#         for i, gs in enumerate(lg):
#             gs = torch.vstack(gs)
#             if self.weight is None:
#                 lg[i] = torch.mean(gs, dim=[0], keepdim=True)
#             else:
#                 gs = gs*self.weight.to(gs.device).reshape(-1,*[1]*(len(gs.shape)-1))
#                 lg[i] = torch.mean(gs, dim=[0], keepdim=True)
#         return lg

class PathIntWeight(_Operation):
    # pathway integrate gradient weight. average from zeros_like(x) to x
    def __init__(self, model, layers, mode='CE', num=30, weight_center=None, guassion_noise=0):
        self.model=model
        self.layers=layers
        self.mode = mode
        self.num = num
        self.weight=None
        if weight_center is not None:
            self.weight = torch.tensor(generate_weight_list(weight_center, num))
        self.guassion_noise=guassion_noise

    def weighted(self, gs):
        return gs * self.weight.to(gs.device).reshape(-1, *[1] * (len(gs.shape) - 1))

    def __call__(self, data):
        x,y = data
        lg = [list() for _ in range(len(self.layers))]
        loss_fun = attack_loss(self.model, self.mode)
        for i in range(self.num):
            ratio = (i + 1) / self.num
            ran = ratio * x
            if self.guassion_noise !=0:
                ran += self.guassion_noise * torch.randn_like(ran)
            ran = ran.detach().requires_grad_()
            loss_fun(ran, y).sum().backward()
            for j, layer in enumerate(self.layers):
                lg[j].append(layer.gradient)
        for i, gs in enumerate(lg):
            gs = torch.vstack(gs)
            if self.weight is not None:
                gs = self.weighted(gs)
            lg[i] = torch.mean(gs, dim=[0], keepdim=True)
        return lg

class NAAWeight(_Wrapper):
    def __init__(self, model, layers, **kwargs):
        super().__init__()
        self.obj = SequenceClass([
            Fixer(PathIntWeight(model, layers, **kwargs)),
        ])


# class LID_Gradient:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#
#     def __call__(self, model, layers, x, y):
#         method = LIDDecomposer(model, **self.kwargs)
#         x = x.detach().requires_grad_()
#         method(toStd(x), y)
#         ag = []
#         for i, layer in enumerate(layers):
#             ag.append(layer.g)
#         return ag
#
# class LIDIG_Gradient:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#
#     def __call__(self, model, layers, x, y):
#         method = LIDIG(model, **self.kwargs)
#         x = x.detach().requires_grad_()
#         method(toStd(x), y)
#         ag = []
#         for i, layer in enumerate(layers):
#             ag.append(layer.gradient.mean(0,True))
#         return ag

# some
class Norm(_Operation):
    def __init__(self, level=1):
        self.level=level
    def __call__(self, g):
        if self.level == 0:
            pass
        elif self.level == 1:
            sum_abs = g.abs().sum([1, 2, 3], True)
            assert not torch.any(sum_abs == 0)
            g = g / sum_abs
        elif self.level == 2:
            sum_sqr = g.square().sum([1, 2, 3], True)
            assert not torch.any(sum_sqr == 0)
            g = g / sum_sqr.sqrt()
        elif self.level == 'inf':
            M = g.abs().max([1, 2, 3], True)
            g = g / M
        return g

class _Updater(_Operation):
    def __init__(self, alpha):
        self.alpha=alpha
    def __call__(self, data):
        x, g = data
        return x - self.alpha * g

# sign: wrong direction
class FGSM_Updater(_Updater):
    def __call__(self, data):
        x, g = data
        return x - self.alpha * g.sign()

class FGNM_Updater(_Updater):
    def __init__(self, alpha):
        super().__init__(alpha)
        self.norm=Norm(2)
    def __call__(self, data):
        x, g = data
        k = g.sign().abs().sum([1, 2, 3], True).sqrt()
        return x - self.alpha * k * self.norm(g)

# combine
# class L2_M11_S:
#     def __init__(self):
#         self.momentum = Momentum(1, 1)
#         self.updater = FGSM_Updater()
#
#     def __call__(self, x, g, alpha):
#         g = norm(g, 2)
#         m = self.momentum(g)
#         x = self.updater(x, m, alpha)
#         return x
#
#     def zeros(self):
#         self.momentum.zeros()
class L2_M11_S(_Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([_Operation(), Norm(2)]),
            ListOperation([_Operation(), Momentum(1, 1)]),
            FGSM_Updater(alpha)
        ])


# class M11_S:
#     def __init__(self):
#         self.momentum = Momentum(1, 1)
#         self.momentum.zeros()
#         self.updater = FGSM_Updater()
#
#     def __call__(self, x, g, alpha):
#         m = self.momentum(g)
#         x = self.updater(x, m, alpha)
#         return x
#
#     def zeros(self):
#         self.momentum.zeros()

class M11_S(_Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([_Operation(), Momentum(1, 1)]),
            FGSM_Updater(alpha)
        ])


# class L2_M11_N:
#     def __init__(self):
#         self.momentum = Momentum(1, 1)
#         self.updater = FGNM_Updater()
#
#     def __call__(self, x, g, alpha):
#         g = norm(g, 2)
#         m = self.momentum(g)
#         x = self.updater(x, m, alpha)
#         return x
#
#     def zeros(self):
#         self.momentum.zeros()
#
#
# class M11_N:
#     def __init__(self):
#         self.momentum = Momentum(1, 1)
#         self.updater = FGNM_Updater()
#
#     def __call__(self, x, g, alpha):
#         m = self.momentum(g)
#         x = self.updater(x, m, alpha)
#         return x
#
#     def zeros(self):
#         self.momentum.zeros()
#
#
# class M11_L2:
#     def __init__(self):
#         self.momentum = Momentum(1, 1)
#         self.updater = GD_Updater()
#
#     def __call__(self, x, g, alpha):
#         m = self.momentum(g)
#         m = norm(m, 2)
#         x = self.updater(x, m, alpha)
#         return x
#
#     def zeros(self):
#         self.momentum.zeros()


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
                 weight_class=None,  # weights
                 activation_class=None,
                 updater_class=None,  # momentum
                 **kwargs):
        self.device = 'cuda'
        self.model = model
        self.eps = eps  # bound
        self.steps = steps  # iterate
        # self.alpha = alpha  # given alpha
        if alpha is None:  # auto alpha
            alpha = self.eps / self.steps
        layers = auto_hook(model, layer_names)  # activations
        self.weight_obj = weight_class(model, layers)  # weights
        self.activation_obj = activation_class(model, layers)
        self.updater = updater_class(alpha)

    def __call__(self, images, labels):
        adv_images = images.clone().detach().to(self.device).requires_grad_()   # [0,1]. un-normalized
        labels = labels.clone().detach().to(self.device)
        self.weight_obj.zeros()
        self.activation_obj.zeros()
        self.updater.zeros()
        for i in range(self.steps):
            lg = self.weight_obj([adv_images, labels])
            la = self.activation_obj(adv_images)
            loss = sum((a * g).sum() for a, g in zip(la, lg))  # layer-wise weighted-activations loss
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = self.updater([adv_images, grad])
            adv_images = torch.clamp(adv_images, min=images - self.eps, max=images + self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images.detach()

    def __del__(self):
        clearHooks(self.model)
