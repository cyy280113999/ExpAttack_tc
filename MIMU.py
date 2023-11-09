import torch
import torch.nn as nn
import torchattacks as ta

# MIM un-targeted
class MIMU(ta.Attack):
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("MIMU", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)

        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            labels = outputs.argmax(dim=1)
            # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
