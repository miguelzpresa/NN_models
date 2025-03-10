import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as utils
import torchvision.models
import fastai.basics as fai
from fastai.callback.hook import hook_outputs



class PerceptualLoss(nn.Module):
    def __init__(self, model, layer_ids, weights):
        super().__init__()
        self.model = model
        self.important_layers = [self.model[i] for i in layer_ids]
        self.hooks = hook_outputs(self.important_layers, detach=False)
        self.weights = weights

    def extract_features(self, x, clone=False):
        self.model(x)
        features = list(self.hooks.stored)
        
        if clone:
            features = [f.clone() for f in features]
        
        return [x]+features
    
    def forward(self, input, target):
        criterion = F.l1_loss
        
        input_features = self.extract_features(input)
        target_features = self.extract_features(target, clone=True)
        
        self.feat_losses = [criterion(in_f, targ_f)*w for in_f, targ_f, w in zip(input_features, target_features, self.weights)]
        
        return sum(self.feat_losses)
    
    def __del__(self): 
        self.hooks.remove() # necesario para que deje de guardar las cosas

def perceptual_loss(weights=[1,2,5,10,20,10]):
    vgg = torchvision.models.vgg16_bn(pretrained=True)
    vgg.eval()
    vgg = vgg.features
    for p in vgg.parameters(): p.requires_grad_(False)
    good_blocks = [i for i,o in enumerate(vgg.children()) if isinstance(o,nn.MaxPool2d)]
    return PerceptualLoss(vgg, layer_ids=good_blocks, weights=weights)
