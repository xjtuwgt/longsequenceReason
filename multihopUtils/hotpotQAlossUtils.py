from torch import nn
import torch
from torch import Tensor as T
import torch.nn.functional as F
from torch.autograd import Variable

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 1.0],  gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-7

    def forward(self, scores: T, targets: T, target_len: T=None):
        prob = torch.sigmoid(scores)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (targets == 1).float()
        if target_len is not None:
            pos_mask = pos_mask.masked_fill(mask=target_len == 0, value=0)
        neg_mask = (targets == 0).float()
        if target_len is not None:
            neg_mask = neg_mask.masked_fill(mask=target_len == 0, value=0)

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask
        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()

        num_neg = 1 if num_neg == 0 else num_neg
        num_pos = 1 if num_pos == 0 else num_pos

        loss = pos_loss / num_pos + neg_loss / num_neg
        return loss

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PairwiseCEFocalLoss(nn.Module):
    def __init__(self, alpha=1.0,  gamma=2.0, reduction='mean'):
        super(PairwiseCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-7
        self.reduction = reduction

    def forward(self, scores: T, targets: T, target_len: T=None):
        batch_size, sample_size = scores.shape #target: 0, 1 tensor
        mask_flag = targets.masked_fill(mask=target_len == 0, value=-1) ## postive=1, negative=0 and mask=-1
        loss = None
        pair_num = 0
        for idx in range(batch_size):
            pos_idx = (mask_flag[idx] >= 1).nonzero(as_tuple=False).squeeze() ## support sentence label (0, 1, 2)
            neg_idx = (mask_flag[idx] == 0).nonzero(as_tuple=False).squeeze()
            if len(pos_idx.shape) == 0:
                pos_idx = torch.tensor([pos_idx]).to(scores.device)
            if len(neg_idx.shape) == 0:
                neg_idx = torch.tensor([neg_idx]).to(scores.device)
            # print(pos_idx, pos_idx.shape, neg_idx.shape)
            pos_num, neg_num = pos_idx.shape[0], neg_idx.shape[0]
            if pos_num > 0 and neg_num > 0:
                pair_num = pair_num + pos_num * neg_num
                pos_scores = scores[idx][pos_idx]
                neg_scores = scores[idx][neg_idx]
                pos_scores = pos_scores.unsqueeze(dim=-1).repeat([1, neg_num])
                neg_scores = neg_scores.unsqueeze(dim=0).repeat([pos_num, 1])
                pair_scores = torch.stack([pos_scores, neg_scores], dim=-1)
                loss_i = self.focal_loss(scores=pair_scores)
                if loss is None:
                    loss = loss_i
                else:
                    loss = loss + loss_i
        if loss is None:
            loss = torch.tensor(0.0).to(scores.device)
        if self.reduction == 'mean':
            loss = loss / batch_size
        else:
            loss = loss
        return loss

    def focal_loss(self, scores: T):
        logpt = F.log_softmax(scores, dim=-1).to(scores.device)
        logpt = logpt[:, :, 0] if len(scores.shape) == 3 else logpt[:, 0]
        pt = Variable(torch.exp(logpt).to(scores.device))
        pt = torch.clamp(pt, self.smooth, 1.0 - self.smooth)
        loss = -self.alpha * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MultiClassFocalLoss(nn.Module):
    def __init__(self, num_class: int, alpha=0.75, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-7
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([1-alpha] + [alpha] * (num_class - 1)) ## class 0 is majority class
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, scores: T, target: T):
        if isinstance(self.alpha, T):
            self.alpha = self.alpha.to(scores.device)
        logpt = F.log_softmax(scores, dim=-1).to(scores.device)
        logpt = logpt.gather(1, target.unsqueeze(dim=-1)).squeeze(dim=-1)
        pt = Variable(torch.exp(logpt).to(scores.device))
        pt = torch.clamp(pt, self.smooth, 1.0 - self.smooth)
        at = Variable(self.alpha.gather(0, target).to(scores.device))
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TriplePairwiseCEFocalLoss(nn.Module):
    def __init__(self, alpha=1.0,  gamma=2.0, reduction='mean'):
        super(TriplePairwiseCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-7
        self.reduction = reduction

    def forward(self, scores: T, head_position: T, tail_position: T, score_mask: T):
        batch_size, seq_num = scores.shape
        if len(head_position.shape) > 1:
            head_position = head_position.squeeze(dim=-1)
        if len(tail_position.shape) > 1:
            tail_position = tail_position.squeeze(dim=-1)
        batch_idx = torch.arange(0, batch_size).to(scores.device)
        score_mask[batch_idx, head_position] = -1
        score_mask[batch_idx, tail_position] = -1
        loss = None
        for idx in range(batch_size):
            neg_idx = (score_mask[idx] == 1).nonzero(as_tuple=False).squeeze()
            if len(neg_idx.shape) == 0:
                neg_idx = torch.tensor([neg_idx]).to(scores.device)
            neg_num = neg_idx.shape[0]
            # print(neg_num)
            if neg_num > 0:
                pos_score = scores[idx][tail_position[idx]].view(1).repeat([neg_num])
                neg_score = scores[idx][neg_idx]
                pair_scores = torch.stack([pos_score, neg_score], dim=-1)
                # print('Pair score = {}'.format(pair_scores))
                loss_i = self.focal_loss(scores=pair_scores)
                if loss is None:
                    loss = loss_i
                else:
                    loss = loss + loss_i
        if loss is None:
            loss = torch.tensor(0.0).to(scores.device)
        if self.reduction == 'mean':
            loss = loss / batch_size
        else:
            loss = loss
        return loss

    def focal_loss(self, scores: T):
        logpt = F.log_softmax(scores, dim=-1).to(scores.device)
        logpt = logpt[:, :, 0] if len(scores.shape) == 3 else logpt[:, 0]
        pt = Variable(torch.exp(logpt).to(scores.device))
        pt = torch.clamp(pt, self.smooth, 1.0 - self.smooth)
        loss = -self.alpha * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt
        if self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

if __name__ == '__main__':
    x = torch.rand((4,10))
    head_position = torch.randint(0, 10, size=(4,1))
    tail_position = torch.randint(0, 10, size=(4,1))
    print(head_position)
    print(tail_position)
    mask = torch.ones((4,10))
    triple_score_loss = TriplePairwiseCEFocalLoss()

    loss = triple_score_loss.forward(scores=x, head_position=head_position, tail_position=tail_position, score_mask=mask)
    print(loss)
    x[torch.arange(0,4), head_position.squeeze(dim=1)] = 100
    loss = triple_score_loss.forward(scores=x, head_position=head_position, tail_position=tail_position,
                                     score_mask=mask)
    print(loss)

    x[torch.arange(0,4), tail_position.squeeze(dim=1)] += 2
    loss = triple_score_loss.forward(scores=x, head_position=head_position, tail_position=tail_position,
                                     score_mask=mask)
    print(loss)

    print()