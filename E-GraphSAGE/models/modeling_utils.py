"""
@Author：Jinfan Zhang
@email：jzhan665@uottawa.ca
@Desc: adversarial training, focal loss
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class PGD(nn.Module):
    def __init__(self,
                 eps=0.05,
                 alpha=0.01,
                 steps=10,
                 mode="targeted",
                 n_classes=None
                 ):
        super().__init__()
        if mode == "targeted":
            assert n_classes is not None
        self.n_classes = n_classes
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.mode = mode

    @torch.no_grad()
    def get_target_labels(self, labels, pred):
        target_labels = torch.zeros_like(labels)
        top_indices = torch.topk(pred, 2)[1]

        for idx in range(labels.shape[0]):
            lab_true = labels[idx]
            choosen_label = top_indices[idx][0] if lab_true != top_indices[idx][0] else top_indices[idx][1]
            target_labels[idx] = choosen_label

        return target_labels.long().cpu()

    def forward(self, inputs, labels, pred=None, model=None):
        model.eval()
        x = inputs.copy()
        x = model.get_embedding(x).cpu()
        perturbation = (torch.rand(1) - 0.5) * self.eps * 2
        perturbation = perturbation.cpu()

        x_ = x + perturbation

        labels = labels.clone().detach().cpu().squeeze()

        if self.mode == "targeted":
            target_labels = self.get_target_labels(labels, pred=pred)
        else:
            target_labels = labels

        loss = nn.CrossEntropyLoss().cpu()

        adv_inputs = x_.clone().detach()
        for _ in range(self.steps):
            adv_inputs.requires_grad = True
            adv_outputs = model._forward(adv_inputs)

            # Calculate loss
            if self.mode == "targeted":
                cost = - loss(adv_outputs, target_labels)
            else:
                cost = loss(adv_outputs, target_labels)

            # Update adversarial samples
            grad = torch.autograd.grad(cost, adv_inputs,
                                       retain_graph=False, create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs - x, min=-self.eps, max=self.eps)
            adv_inputs = torch.clamp(x + delta, min=0, max=1).detach()

        return adv_inputs


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)
        return loss

