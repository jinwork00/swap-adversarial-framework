import torch
from torch import nn
from .components import (EEGFeatureExtractor, ChannelNormLayer,
                         DepthwiseConv2d, PointwiseConv2d)
from ..utils.domain_adaptation import GradientReversal
from .losses import negative_entropy

class EEGNetDAL(nn.Module):

    def __init__(self, nb_classes=2, domain_classes=3, norm_rate=0.25,
                 Chans=64, Samples=128, kernLength=64, F1=8, D=2, F2=16,
                 grl_lambda=0.3, mi_lambda=0.01, dropoutType='Dropout',
                 dropoutRate=0.5, class_weight=None, mi_lambda_learnable=False,
                 channel_norm=True, domain_label_smoothing=0.1):
        super().__init__()
        if Chans < 1 or Samples < 32:
            raise ValueError('Chans must be positive and Samples must be >= 32')
        self.norm_rate = norm_rate
        self.grl_lambda = grl_lambda
        self.mi_lambda_learnable = mi_lambda_learnable
        self.mi_lambda = (nn.Parameter(torch.full((nb_classes,), float(mi_lambda)))
                           if mi_lambda_learnable else float(mi_lambda))
        self.channel_norm_layer = ChannelNormLayer(norm_rate) if channel_norm else None
        self.grl = GradientReversal(grl_lambda)
        self.domain_softmax = nn.Softmax(dim=-1)
        weights = torch.tensor(class_weight, dtype=torch.float) if class_weight is not None else None
        self.target_criterion = nn.CrossEntropyLoss(weight=weights)
        self.domain_criterion = nn.CrossEntropyLoss(label_smoothing=domain_label_smoothing)
        self.feature_extractor = EEGFeatureExtractor(
            Chans=Chans, Samples=Samples, dropoutRate=dropoutRate,
            kernLength=kernLength, F1=F1, D=D, F2=F2, dropoutType=dropoutType)
        self.target_classifier = nn.Sequential(nn.Flatten(), nn.Linear(F2*(Samples//32), nb_classes))
        self.domain_classifier = nn.Sequential(nn.Flatten(), nn.Linear(F2*(Samples//32), domain_classes))
        self.apply(self.weight_init)

    def first_step_loss(self, target_logits, domain_logits, target_labels):

        target_loss = self.target_criterion(target_logits, target_labels)
        entropy_loss = negative_entropy(self.domain_softmax(domain_logits),
                                        self.mi_lambda, target_labels)
        return target_loss + entropy_loss

    def domain_loss(self, domain_logits, domain_labels):

        return self.domain_criterion(domain_logits, domain_labels)

    def forward(self, x, second_step=False):
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.channel_norm_layer is not None:
            x = self.channel_norm_layer(x)
        features = self.feature_extractor(x)
        target_output = self.target_classifier(features)
        if second_step:
            features = self.grl(features)
        domain_output = self.domain_classifier(features)
        return (target_output, domain_output)

    def weight_init(self, m):
        if isinstance(m, DepthwiseConv2d) or isinstance(m, PointwiseConv2d):
            nn.init.xavier_uniform_(m.conv.weight)
            if isinstance(m, DepthwiseConv2d):
                with torch.no_grad():
                    norm = m.conv.weight.data.norm(2, dim=(1, 2, 3), keepdim=True)
                    desired = torch.clamp(norm, max=1.0)
                    m.conv.weight.data *= desired / (1e-06 + norm)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            with torch.no_grad():
                norm = m.weight.data.norm(2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=self.norm_rate)
                m.weight.data *= desired / (1e-06 + norm)

    def apply_max_norm(self):
        with torch.no_grad():
            weight = self.target_classifier[1].weight
            norm = weight.data.norm(2, dim=1, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            weight.data *= desired / (1e-06 + norm)

