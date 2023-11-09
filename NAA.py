import torch
import torch.nn as nn
import torchattacks as ta
import numpy as np
from utils import *

class NAA(ta.Attack):
    def __init__(self, model, layer_name, ag=True,gn=False,ug=False, **kwargs):
        super().__init__("NAA", model)
        self.eps = 16/255
        self.steps = 10
        self.alpha = 1.6 / 255
        self.decay = 1.0
        self.supported_mode = ["default", "targeted"]
        self.layer = auto_hook(model,layer_name)[0]  # single layer
        self.aggregate_gradient=ag
        self.gaussian_noise=gn
        self.update_gradient=ug
        self._momentum=None

    def momentum(self, g=None):
        if g is None:  # clear
            self._momentum = None
        elif self._momentum is None:  # init
            self._momentum = g
        else:
            self._momentum = g + self._momentum * self.decay
        return self._momentum

    def forward(self, images, labels, **kwargs):
        images = images.clone().detach().to(self.device)  # [0,1]
        labels = labels.clone().detach().to(self.device)

        self.momentum(None)
        self.momentum(torch.zeros_like(images).detach().to(self.device))

        adv_images = images.clone()
        pred_fun = lambda x: self.get_logits(x).softmax(1)[:, labels]
        for i in range(self.steps):
            adv_images = adv_images.detach().requires_grad_()
            if i == 0 or self.update_gradient:
                ag = []
                for i in range(30):
                    ratio=(i+1)/10
                    ran = ratio * adv_images
                    ran = ran.detach().requires_grad_()
                    pred_fun(ran).sum().backward()
                    ag.append(self.layer.gradient)
                g = torch.mean(torch.vstack(ag), dim=[0], keepdim=True)
            adv_images.grad = None
            outputs = self.get_logits(adv_images)
            cost = (self.layer.activation * g).sum()  # minimize FIA loss

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = self.momentum(grad)
            adv_images = adv_images.detach() - self.alpha * momentum.sign()  # apply "-" to minimize loss
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1)
        return adv_images.detach()

    def __del__(self):
        clearHooks(self.model)
