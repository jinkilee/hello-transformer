import torch
from torch.nn.modules.loss import _Loss


class Criterion(_Loss):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target, return_pred_label=False):  # (Q,C) (Q)
        target = target[self.amount:]
        target_onehot = torch.zeros_like(probs)
        #print('** sum of probs/target_onehot: {} {}'.format(probs.sum(), target_onehot.sum()))
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        acc = torch.sum(target == pred).float() / target.shape[0]
        #print('** acc: {}'.format(acc))

        if return_pred_label:
            return loss, acc, pred, target
        else:
            return loss, acc
