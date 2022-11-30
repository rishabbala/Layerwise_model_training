import torch
from torch import nn



class ContrastiveLoss(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.temperature = 0.1
        self.base_temperature = 0.07
        self.device = device


    def forward(self, features, labels):

        # features = features.view(features.shape[0], -1)

        # labels = labels.view(-1, 1)
        # mask = torch.eq(labels, labels.T).float()

        # f_dot_f = torch.matmul(features, features.T)/self.temperature
        # f_max, _ = torch.max(f_dot_f, dim=1, keepdim=True)
        # f_dot_f = f_dot_f - f_max.detach() ## for stability from the supcl github

        # logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=self.device)
        # mask = mask * logits_mask

        # exp_f_dot_all = torch.exp(f_dot_f) * logits_mask
        # log_prob = f_dot_f - torch.log(exp_f_dot_all.sum(1, keepdim=True))

        # loss = - torch.sum((mask * log_prob), dim=1) / torch.sum(mask, dim=1)
        # loss = loss.mean()
        
        # return loss


        features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
