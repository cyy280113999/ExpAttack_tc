import torch
import numpy as np
from utils import *
from methods.LIDDecomposer import LIDDecomposer
from methods.LIDIG import LIDIG


device='cuda'


class Operation:
    # build customized attack method.
    def __call__(self, x):
        return x  # pass
    def zeros(self):
        pass

class Wrapper(Operation):
    # will apply a single self.obj to input
    def __call__(self, x):
        return self.obj(x)
    def zeros(self):
        if hasattr(self.obj,'zeros'):
            self.obj.zeros()

class ListOperation(Operation):
    # will apply a list of operations to a list of inputs
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

class SequenceClass(Operation):
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

# utils
class Fixer(Operation):
    # output the input at the first calling time in memory
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
            self.mem = self.op(x)  # keep first input
            self.fix=True
        return self.mem


class Momentum(Operation):
    # momentum accumulate the input
    def __init__(self, decay=0.9, keep=0.1):
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

class WeightedSum(Operation):
    def __init__(self, weights):
        self.weights=weights
    def __call__(self, xs):
        return sum(x*w for x,w in zip(xs,self.weights))

class Switch(Operation):
    #  output first firstly, second secondly
    def __init__(self, op1, op2):
        self.op1=op1
        self.op2=op2
        self.switch = False
    def zeros(self):
        self.switch = False
        if hasattr(self.op1, 'zeros'):
            self.op1.zeros()
        if hasattr(self.op2, 'zeros'):
            self.op2.zeros()
    def __call__(self, x):
        if not self.switch:
            self.switch=True
            return self.op1(x)
        else:
            return self.op2(x)
# input
class DIM(Operation):
    def __init__(self, image_size=224, image_resize=299, prob=0.7):
        self.image_size = image_size
        self.image_resize = image_resize
        self.prob = prob

    def __call__(self, input_tensor):
        rnd = torch.randint(self.image_size, self.image_resize, ())
        rescaled = nf.interpolate(input_tensor, size=(rnd, rnd), mode='nearest')
        h_rem = self.image_resize - rnd
        w_rem = self.image_resize - rnd
        pad_top = torch.randint(0, h_rem, ())
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_rem, ())
        pad_right = w_rem - pad_left
        padded = nf.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0.)
        padded = padded[:, :, :self.image_resize, :self.image_resize]
        ret = torch.where(torch.rand(1) < self.prob, padded, input_tensor)
        ret = nf.interpolate(ret, size=(self.image_size, self.image_size), mode='nearest')
        return ret
# activation implement
class LayerActivation(Operation):
    def __init__(self, model, layers):
        self.model=model
        self.layers=layers
    def __call__(self, x):
        la = []
        self.model(toStd(x))
        for l in self.layers:
            la.append(l.activation)
        return la

class FixActivation(Wrapper):
    def __init__(self, model, layers):
        super().__init__()
        self.obj=SequenceClass([
            Fixer(LayerActivation(model, layers)),
        ])


class Diff_Activation(Operation):
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

class LayerWeight(Operation):
    def __init__(self, model, layers, mode='CE'):
        self.model=model
        self.layers=layers
        self.mode=mode
    def __call__(self, data):
        x, y = data  # weight requires x and y
        x=x.detach().requires_grad_()
        loss_fun = attack_loss(self.model, self.mode)
        loss_fun(x, y).sum().backward()
        lg = []
        for layer in self.layers:
            lg.append(layer.gradient)
        return lg

class NoWeight(Wrapper):
    # set weight all ones
    def __init__(self, model, layers):
        self.model=model
        self.layers=layers
        self.weight=None
    def __call__(self, data):
        if self.weight is None:
            x, y = data
            self.model(toStd(x))
            self.weight = []
            for layer in self.layers:
                self.weight.append(torch.ones_like(layer.gradient))
        return self.weight

class FeatureMomentumWeight(Wrapper):
    def __init__(self, model, layers, decay=0.9, keep=0.1, **kwargs):
        self.obj = SequenceClass([
            LayerWeight(model, layers, **kwargs),
            ListOperation([Momentum(decay, keep)for _ in range(len(layers))]),
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

class BinRanWeight(Operation):
    # binomial random average gradient
    def __init__(self, model, layers, mode='CE', prob=0.9, num=30):
        self.model=model
        self.layers=layers
        self.mode = mode
        self.prob = prob  # keep prob. as to 1 - drop prob
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
class FIAWeight(Wrapper):
    def __init__(self, model, layers, **kwargs):
        self.obj = Fixer(BinRanWeight(model, layers, **kwargs))

class FIAMWeight(Wrapper):
    def __init__(self, model, layers, decay=0.9, keep=0.1, **kwargs):
        self.obj = SequenceClass([
            BinRanWeight(model, layers, **kwargs),
            ListOperation([Normalize(1)]),
            ListOperation([Momentum(decay, keep)for _ in range(len(layers))]),
        ])

class FMAAWeight(Wrapper):
    def __init__(self, model, layers, decay=0.9, keep=0.1, **kwargs):
        prob1 = kwargs.pop('prob1')
        prob2 = kwargs.pop('prob2')
        self.obj = SequenceClass([
            Switch(op1=BinRanWeight(model, layers, prob=prob1,**kwargs),
                   op2=BinRanWeight(model, layers, prob=prob2,**kwargs)),
            ListOperation([Momentum(decay, keep)for _ in range(len(layers))]),
        ])

class FMAA0Weight(Wrapper):
    def __init__(self, model, layers, decay=0.9, keep=0.1, **kwargs):
        self.obj = SequenceClass([
            Switch(op1=BinRanWeight(model, layers,**kwargs),
                   op2=LayerWeight(model, layers,**kwargs)),
            ListOperation([Normalize(1)]),
            ListOperation([Momentum(decay, keep)for _ in range(len(layers))]),
        ])

class GuassRanWeight(Operation):
    def __init__(self, model, layers, mode='CE', num=30, sigma=0.1):
        self.model=model
        self.layers=layers
        self.mode = mode
        self.num=num
        self.sigma=sigma

    def __call__(self, data):
        x,y = data
        lg = [list() for _ in range(len(self.layers))]
        loss_fun = attack_loss(self.model, self.mode)
        for j in range(self.num):
            ran = x + self.sigma * torch.randn_like(x)
            ran = ran.detach().requires_grad_()
            loss_fun(ran, y).sum().backward()
            for i, layer in enumerate(self.layers):
                lg[i].append(layer.gradient)
        for i, gs in enumerate(lg):
            gs = torch.vstack(gs)  # list to tensor
            lg[i] = torch.mean(gs, dim=[0], keepdim=True)
        return lg

# tricks
def gaussian_weight(x, center, sigma):
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def generate_weight_list(center, n, sigma=None):
    if sigma is None:
        sigma = n / 10
    weights = [gaussian_weight(opacity, center, sigma) for opacity in range(n)]
    return weights


class PathIntWeight(Operation):
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

class NAAWeight(Wrapper):
    def __init__(self, model, layers, **kwargs):
        super().__init__()
        self.obj = SequenceClass([
            Fixer(PathIntWeight(model, layers, **kwargs)),
        ])

class NAAMWeight(Wrapper):
    def __init__(self, model, layers, decay=0.9, keep=0.1, **kwargs):
        self.obj = SequenceClass([
            PathIntWeight(model, layers, **kwargs),
            ListOperation([Normalize(2)]),
            ListOperation([Momentum(decay, keep)for _ in range(len(layers))]),
        ])


class HalfWeight(Operation):
    def __init__(self, model, layers, mode='CE', num=30, ratio=0.5):
        self.model=model
        self.layers=layers
        self.mode = mode
        self.num = num
        self.ratio=ratio

    def __call__(self, data):
        x,y = data
        lg = [list() for _ in range(len(self.layers))]
        loss_fun = attack_loss(self.model, self.mode)
        for i in range(self.num):
            ran = x * self.ratio
            ran = ran.detach().requires_grad_()
            loss_fun(ran, y).sum().backward()
            for j, layer in enumerate(self.layers):
                lg[j].append(layer.gradient)
        for i, gs in enumerate(lg):
            gs = torch.vstack(gs)
            lg[i] = torch.mean(gs, dim=[0], keepdim=True)
        return lg

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
class LIDIG_gradient(Operation):
    def __init__(self, model, layers, **kwargs):
        self.method = LIDIG(model, **kwargs)
        self.layers = layers
    def __call__(self, data):
        x, y = data
        x = x.detach().requires_grad_()
        self.method(toStd(x), y)
        ag = []
        for i, layer in enumerate(self.layers):
            ag.append(layer.gradient.mean(0,True))
        return ag
class LIDIGWeight(Wrapper):
    def __init__(self, model, layers, **kwargs):
        self.obj = Fixer(LIDIG_gradient(model, layers, **kwargs))

class TIM:
    def __init__(self, Tkern_size=15):
        self.T_kern = self._gkern(Tkern_size)

    def _gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = np.exp(-(x**2) / (2 * nsig**2))
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 3)
        return torch.from_numpy(stack_kernel)

    def __call__(self, grad):
        return nf.conv2d(grad, self.T_kern, stride=1, padding='same')


def zeroToOne(d):
    check = torch.argwhere(d == 0)
    d[check] = 1
    return d
# some
def norm(x, level=2):
    if level == 0:
        return x.max([1,2,3],True)
    elif level == 1:
        return x.abs().sum([1, 2, 3], True)
    elif level == 2:
        return x.square().sum([1, 2, 3], True).sqrt()
    elif level == 'inf':
        return x.abs().max([1, 2, 3], True)
class Normalize(Operation):
    def __init__(self, level=1):
        self.level=level
    def __call__(self, g):
        g_norm=norm(g)
        g_norm = zeroToOne(g_norm)  # not divide zero
        g = g / g_norm
        return g

class _Updater(Operation):
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
    def __call__(self, data):
        x, g = data
        return x - self.alpha * norm(g.sign(), 2) / zeroToOne(norm(g, 2)) * g


# Example usage:
# k=3
class PIM_Updater(_Updater):
    def __init__(self, alpha, amplification_factor=1., gamma=1.):
        super().__init__(alpha)
        self.amplification_factor = amplification_factor
        self.gamma = gamma
        self.P_kern, self.kern_size = self.project_kern(3)  # 3//2=1

    def project_kern(self, kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern]).swapaxes(0, 2)
        stack_kern = np.expand_dims(stack_kern, 3)
        stack_kern = torch.from_numpy(stack_kern)
        return stack_kern, kern_size // 2

    def project_noise(self, x):
        kern_size=self.kern_size
        x = nf.pad(x, (kern_size, kern_size, kern_size, kern_size))
        self.P_kern = self.P_kern.to(x.device)
        x = nf.conv2d(x, self.P_kern, stride=1, padding=0)
        return x

    def __call__(self, data):
        x, g = data

        # PIM method
        alpha_beta = self.alpha * self.amplification_factor
        gamma = self.gamma * alpha_beta

        amplification_update = alpha_beta * g.sign()
        cut_noise = torch.clamp(torch.abs(amplification_update) - self.alpha, 0.0, 10000.0) * torch.sign(amplification_update)
        projection = gamma * torch.sign(self.project_noise(cut_noise))

        amplification_update += projection

        return x - amplification_update

# combine
class L1_M11_S(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Normalize(1)]),
            ListOperation([Operation(), Momentum(1, 1)]),
            FGSM_Updater(alpha)
        ])
class L1_S(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Normalize(1)]),
            FGSM_Updater(alpha)
        ])
class L2_M11_S(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Normalize(2)]),
            ListOperation([Operation(), Momentum(1, 1)]),
            FGSM_Updater(alpha)
        ])
class M11_S(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Momentum(1, 1)]),
            FGSM_Updater(alpha)
        ])

class L1_M11_N(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Normalize(1)]),
            ListOperation([Operation(), Momentum(1, 1)]),
            FGNM_Updater(alpha)
        ])
class M11_N(Wrapper):
    def __init__(self, alpha):
        self.obj=SequenceClass([
            ListOperation([Operation(), Momentum(1, 1)]),
            FGNM_Updater(alpha)
        ])

def vector_preprocess(x):
    return x.reshape(1,-1), x.shape

def orthogonal_decomposition(decomped, direction):
    shape = decomped.shape
    decomped = decomped.reshape(1, -1)
    direction = direction.reshape(1, -1)
    direction = direction/zeroToOne(direction.sum([1],True).sqrt())
    parall_len = (decomped * direction).sum([1],True)
    paralleled = (parall_len * direction).reshape(shape)
    orthogonal = (decomped - paralleled).reshape(shape)
    return orthogonal, paralleled

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

    def __init__(self, model=None,
                 layers=None,  # hooked
                 activation_class=None,  # activations
                 weight_class=None,  # weights
                 steps=10, alpha=None,  # iterate
                 updater_class=None,  # momentum
                 eps=8 / 255,  # bound
                 channel_wise=None,
                 DIM_caller=None,
                 **kwargs):
        self.model = model
        self.layers = layers
        self.eps = eps  # bound
        self.steps = steps  # iterate
        # self.alpha = alpha  # given alpha
        if alpha is None:  # auto alpha
            alpha = self.eps / self.steps
        self.weight_obj = weight_class(model, layers)  # weights
        self.activation_obj = activation_class(model, layers)
        self.updater = updater_class(alpha)
        self.channel_wise=channel_wise
        self.noise_loss=None
        self.DIM_caller=DIM_caller

    def __call__(self, images, labels):
        images = images.to(device)  # [0,1]. un-normalized
        adv_noise = torch.zeros_like(images).requires_grad_()
        labels = labels.clone().detach().to(device)
        self.weight_obj.zeros()
        self.activation_obj.zeros()
        self.updater.zeros()
        adv_images = images + adv_noise
        adv_images = torch.clamp(adv_images, min=0, max=1)
        for i in range(self.steps):
            if self.DIM_caller is not None:
                adv_images = self.DIM_caller(adv_images)
            la = self.activation_obj(adv_images)
            lg = self.weight_obj([adv_images, labels])
            if self.channel_wise:
                lg = [g.mean([2, 3], True) for g in lg]
                la = [a.mean([2, 3], True) for a in la]
            loss = sum((a * g).sum() for a, g in zip(la, lg))  # layer-wise weighted-activations loss
            if self.noise_loss is not None:
                loss += 0
            grad = torch.autograd.grad(loss, adv_noise, retain_graph=False, create_graph=False)[0]
            adv_noise = self.updater([adv_noise, grad])
            adv_noise = torch.clamp(adv_noise, min=-self.eps, max=self.eps)
            adv_images = images + adv_noise
            adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_noise.detach()

