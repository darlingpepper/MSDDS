import torch
import torch.nn as nn
import torch.nn.functional as F
def save_model(model, save_name):
    path = f'output/{save_name}'
    torch.save(model.state_dict(), path)
def load_model(model, load_name):
    path = f'output/{load_name}'
    model.load_state_dict(torch.load(path))
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def compute_kl_loss(p, q, pad_mask=None):
	p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
	q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
	if pad_mask is not None:
		p_loss.masked_fill_(pad_mask, 0.)
		q_loss.masked_fill_(pad_mask, 0.)
	p_loss = p_loss.mean()
	q_loss = q_loss.mean()
	loss = (p_loss + q_loss) / 2
	return loss


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, *views, labels=None, mask=None):
        features = self.get_forward_features_from_multi_view(*views)  # shape [bsz, n_views, ...].

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

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
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
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
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def get_forward_features_from_multi_view(self, *views):
        '''

        :param views: the shape for each view must be same: [batch, h_dim]
        :return: features, which can be input to function forward().
        '''
        assert len(views) >= 2, "the number of view must >= 2."
        for idx in range(len(views)):
            assert views[0].shape == views[idx].shape
        features = []
        for idx in range(len(views)):
            features.append(F.normalize(views[idx], dim=1).unsqueeze(1))
        features = torch.cat(features, dim=1)
        return features
    


def cal_cl_loss(drug1_id, out1_2d, out1_3d, drug2_id, out2_2d, out2_3d):
    temperature=0.07
    base_temperature=0.07
    cl_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
    loss = (cl_loss(out1_2d, out1_3d, labels=drug1_id) + cl_loss(out2_2d, out2_3d, labels=drug2_id)) / 2
    return loss