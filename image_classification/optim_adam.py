import torch
import numpy as np


class MAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2_range=(0.5, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, ams_warmup_steps=1,
                 ams_reset_freq=-1, use_sgd=False, moment_warmup=0, neq_thresh=0,
                 nesterov=False, adamw=True, share_beta=False, lamb=False, max_grad_norm=0):
        defaults = dict(lr=lr, beta1=beta1, beta2_range=beta2_range, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,ams_warmup_steps=ams_warmup_steps,
                        ams_reset_freq=ams_reset_freq, use_sgd=use_sgd, moment_warmup=moment_warmup,
                        neq_thresh=neq_thresh, nesterov=nesterov, adamw=adamw, share_beta=share_beta,
                        lamb=lamb)
        self.max_grad_norm = max_grad_norm
        self.last_step = 0
        super(MAdam, self).__init__(params, defaults)

    def get_beta(self):
        if len(self.adaptive_beta) > 0:
            return np.max(self.adaptive_beta)
        else:
            return None

    @property
    def update_size(self):
        if getattr(self, "update_size_", None) is not None:
            return None, None, self.update_size_
        else:
            return None, None, None

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.update_size_ = None

        grad_all_32 = []

        total_grad_norm = 0
        for group in self.param_groups:
            for p in group['params']:
                grad_all_32.append(p.grad.data.float())
                total_grad_norm += grad_all_32[-1].square().sum()
        total_grad_norm = torch.sqrt(total_grad_norm)

        if self.max_grad_norm > 0:
            clipped_ratio = self.max_grad_norm / max(self.max_grad_norm, total_grad_norm)
        else:
            clipped_ratio = 1.

        gidx = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = grad_all_32[gidx] * clipped_ratio
                gidx += 1
                # grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                amsgrad = group['amsgrad']

                p_data_fp32 = p.data.float()

                if not group['adamw'] and group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p_data_fp32)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_grad'] = torch.zeros_like(p_data_fp32)
                    state['total_w'] = torch.zeros_like(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1 = group['beta1']
                beta2_min, beta2_max = group['beta2_range']

                state['step'] += 1
                total_w = state['total_w']

                if state['step'] == 1 or beta2_max == beta2_min:
                    exp_avg.mul_(beta2_max).add_(1-beta2_max, grad)
                    exp_avg_sq.mul_(beta2_max).addcmul_(1-beta2_max, grad, grad)
                    state['total_w'] = 1 - beta2_max ** state['step']
                else:
                    # find the beta that maximize the variance
                    # beta is the multiplier for the new grad
                    exp_avg_sq_unbiased = exp_avg_sq / total_w
                    exp_avg_unbiased = exp_avg / total_w
                    moment_diff = exp_avg_sq_unbiased - exp_avg_unbiased ** 2
                    mean_diff_sq = (grad - exp_avg_unbiased) ** 2
                    sum_diff = mean_diff_sq + moment_diff
                    denominator = (mean_diff_sq - moment_diff).mul_(total_w).add_(sum_diff)

                    adv_beta = sum_diff.div_(denominator.add_(1e-16))

                    # clamp the range
                    adv_beta.clamp_(min=beta2_min, max=beta2_max)

                    adv_beta_comp = 1 - adv_beta
                    exp_avg.mul_(adv_beta).add_(adv_beta_comp * grad)
                    exp_avg_sq.mul_(adv_beta).add_(adv_beta_comp.mul(grad).mul_(grad))

                    state['total_w'] = state['total_w'] * adv_beta + adv_beta_comp

                if state['step'] <= group['moment_warmup']:
                    continue

                denom = (exp_avg_sq / state['total_w']).sqrt() + group['eps']

                if amsgrad:
                    torch.max(denom, max_exp_avg_sq, out=max_exp_avg_sq)
                    denom.copy_(max_exp_avg_sq)

                state['exp_avg_grad'].mul_(beta1).add_(grad, alpha=(1 - beta1))

                bias_correction0 = 1 - beta1 ** (state['step'] - group['moment_warmup'])
                step_size = group['lr'] / bias_correction0

                if group['nesterov']:
                    exp_avg_grad = state['exp_avg_grad'] * beta1 + (1-beta1) * grad
                else:
                    exp_avg_grad = state['exp_avg_grad']

                if group['lamb']:
                    if bias_correction0 < 1:
                        update_ = exp_avg_grad / denom / bias_correction0
                    else:
                        update_ = exp_avg_grad / denom
                    update_.add_(p_data_fp32, alpha=group['weight_decay'])

                    trust_ratio = 1.
                    if group['weight_decay'] > 0:
                        weight_norm = torch.norm(p_data_fp32) #.clamp(0, 10)
                        update_norm = torch.norm(update_)
                        if weight_norm == 0 or update_norm == 0:
                            trust_ratio = 1.
                        else:
                            trust_ratio = weight_norm / update_norm

                    p_data_fp32.add_(update_, alpha=-group['lr']*trust_ratio)
                else:
                    if group['adamw'] and group['weight_decay'] > 0:
                        p_data_fp32.add_(- group['lr'] * group['weight_decay'], p_data_fp32)

                    if False:
                        if group['share_beta']:
                            update = - step_size * exp_avg / denom
                        else:
                            update = - step_size * exp_avg_grad / denom
                        p_data_fp32.add_(update)
                        self.update_size_ = torch.mean(update.abs()).item()
                    else:
                        if group['share_beta']:
                            p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                        else:
                            p_data_fp32.addcdiv_(-step_size, exp_avg_grad, denom)

                p.data.copy_(p_data_fp32)

        return loss



