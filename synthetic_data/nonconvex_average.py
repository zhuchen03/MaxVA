import math
from collections import defaultdict
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)
font = {'family' : "Times New Roman",
        'size'   : 20}
plt.rc('font', **font)
import pdb
import random
import json
import pickle


def grad_func(x, i):
    if i == 1:
        if abs(x) <= 1:
            return 11 * x
        elif x < 0:
            return -11
        else:
            return 11
    else:
        if abs(x) <= 1:
            return - x
        elif x < 0:
            return 1
        else:
            return -1

def func(x):
    if abs(x) <= 1:
        return 5.5 * (x ** 2) - 5 * (x ** 2)
    else:
        return 11 * abs(x) - 5.5 - 10 * abs(x) + 5

class Adam:
    def __init__(self, lr, beta2=0.9, amsgrad=False, eps=1e-8):
        self.lr = lr
        self.beta2 = beta2
        self.amsgrad = amsgrad
        self.max_exp_avg_sq = 0
        self.exp_avg_sq = 0
        self.eps = eps
        self.adaptive_lr = 0
        self.grad_step = 0
        self.steps = 0
        # records
        self.total_change_alr = 0
        self.total_alr = 0
        self.total_astep = 0

    def step(self, x, grad):
        t = self.steps

        self.exp_avg_sq = self.exp_avg_sq * self.beta2 + (1 - self.beta2) * (grad ** 2)
        bc = 1 - self.beta2 ** (t + 1)
        self.steps += 1

        if self.amsgrad:
            self.max_exp_avg_sq = max(self.max_exp_avg_sq, self.exp_avg_sq)
            exp_avg_sq = self.max_exp_avg_sq
        else:
            exp_avg_sq = self.exp_avg_sq

        # the terms to be logged
        adapt_lr = self.lr / (math.sqrt(exp_avg_sq / bc) + self.eps)
        self.total_change_alr += abs(adapt_lr - self.adaptive_lr)
        self.adaptive_lr = adapt_lr
        self.total_alr += adapt_lr
        self.total_astep += adapt_lr * abs(grad)

        x -= adapt_lr * grad
        return x

    def get_metrics(self):
        return self.total_alr, self.total_astep, self.total_change_alr


class MAdam:
    def __init__(self, lr, beta1=0, beta2=0.9, amsgrad=False, eps=1e-8):
        self.lr = lr
        self.beta2_max = beta2
        self.beta1 = beta1
        self.amsgrad = amsgrad
        self.max_exp_avg_sq = 0
        self.exp_avg_sq = 0
        self.exp_avg = 0
        self.w = 0

        self.eps = eps
        self.adaptive_lr = 0
        self.grad_step = 0
        self.steps = 0
        # records
        self.total_change_alr = 0
        self.total_alr = 0
        self.total_astep = 0

    def step(self, x, grad):
        # madam step
        t = self.steps
        bmax = self.beta2_max
        if t == 0:
            if self.beta2_max >= 1:
                bmax = 0.9
            adv_beta = bmax
        else:
            moment_diff = max(self.exp_avg_sq / self.w - (self.exp_avg / self.w) ** 2, 0)
            mean_diff_sq = (grad - self.exp_avg) ** 2
            # w_diff_diff = total_w * (mean_diff_sq - moment_diff)
            sum_diff = mean_diff_sq + moment_diff
            denominator = (mean_diff_sq - moment_diff) * self.w + sum_diff

            adv_beta_noclip = sum_diff / (denominator + 1e-20)
            adv_beta = min(max(0.5, adv_beta_noclip), bmax)

        self.exp_avg_sq = self.exp_avg_sq * adv_beta + (1 - adv_beta) * (grad ** 2)
        self.exp_avg = self.exp_avg * adv_beta + (1 - adv_beta) * grad
        self.w = self.w * adv_beta + (1 - adv_beta)
        bc = self.w
        self.steps += 1

        if self.amsgrad:
            self.max_exp_avg_sq = max(self.max_exp_avg_sq, self.exp_avg_sq)
            exp_avg_sq = self.max_exp_avg_sq
        else:
            exp_avg_sq = self.exp_avg_sq

        # the terms to be logged
        adapt_lr = 1 / (math.sqrt(exp_avg_sq / bc) + self.eps)
        self.total_change_alr += abs(adapt_lr - self.adaptive_lr)
        self.adaptive_lr = adapt_lr
        self.total_alr += adapt_lr
        self.total_astep += adapt_lr * abs(grad)

        x -= adapt_lr * grad
        return x

    def get_metrics(self):
        return self.total_alr, self.total_astep, self.total_change_alr


def update_dict(some_dict, total_alr, total_astep, total_change_alr, grad, x):
    some_dict['total_alr'].append(total_alr)
    some_dict['total_astep'].append(total_astep)
    some_dict['total_clr'].append(total_change_alr)
    some_dict['grad'].append(grad)
    some_dict['fx'].append(x)


def plot_errs(lr_list, err_list, std_list, c1="or", c2="gray", label="MAdam"):
    # plt.plot(lr_list, err_list, c1, label=label)
    plt.plot(lr_list, err_list, '-', color=c2, label=label)

    plt.fill_between(lr_list, err_list - std_list, err_list + std_list,
                     color=c2, alpha=0.2)


def plot_curves(opt_dict_dict, plot_key, ylabel, downsample=10, ylim_min=-1., ylim_max=-1.):
    color_dict = {"LaMAdam": "C0",
                  "LaProp": "C1",
                  "Adam": "C2",
                  "MAdam": "C3",
                  "AMSGrad": "C4",
                  }

    line_style_dict = {"LaMAdam": "-",
                       "LaProp": "-",
                       "Adam": "-",
                       "MAdam": "-",
                       "AMSGrad": "-",
                       }
    fig, ax = plt.subplots()
    keys = ["Adam", "AMSGrad", "MAdam"]
    # for key, opt_dict in opt_dict_dict.items():
    for key in keys:
        opt_dict = opt_dict_dict[key]
        res_arr_list = []
        for res_dict in opt_dict:
            res_arr_list.append(np.array(res_dict[plot_key][::downsample]))
        res_arr = np.stack(res_arr_list)
        res_median = np.median(res_arr, axis=0)
        res_std_err = np.std(res_arr, axis=0) / np.sqrt(res_arr.shape[0])
        iter_nums = np.arange(len(res_median)) * downsample+1
        plot_errs(iter_nums, res_median, res_std_err, c2=color_dict[key], label=key)

    plt.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if ylim_min>0:
        ax.set_ylim(bottom=ylim_min, ymax=ylim_max)#, ymax=ymax)
    # else:
    #     ax.set_ylim(ymax=ymax)
    # plt.savefig("plots/nonconvex-{}-average.pdf".format(plot_key), bbox_inches='tight')
    plt.savefig("plots/nonconvex-{}-average.eps".format(plot_key), format='eps', bbox_inches='tight')



def grid_search(min_lr, max_lr, incre, x_init, Opt, beta, iters):
    min_loss = float("inf")
    for lr in np.arange(min_lr, max_lr, incre):
        x = x_init
        opt = Opt(lr=lr, beta2=beta)
        for i in range(iters):
            sidx = random.randint(1, 11)
            grad = grad_func(x, sidx)
            x = opt.step(x, grad)
        if abs(x) < min_loss:
            if abs(x) < min_loss:
                min_loss = abs(x)
                best_lr = lr
    print(min_loss)
    return best_lr

if __name__ == "__main__":
    # lr = 1 #* 1.5
    resume = "results/nonconvex_results_average.pkl"
    if resume:
        # Allows to modify the plot style without re-run the experiments
        # opt_dict_dict = json.load(open(resume))
        opt_dict_dict = pickle.load(open(resume, 'rb'))
        # plot_curves(opt_dict_dict, "total_alr", r"$\sum_{t=1}^{T} 1/\sqrt{v_t}$")
        # plot_curves(opt_dict_dict, "total_astep", r"$\sum_{t=1}^{T}|| g_t/\sqrt{v_t}||^2$")
        # plot_curves(opt_dict_dict, "total_clr",
        #             r"$\sum_{t=1}^{T}|| \frac{\eta_{t}}{\sqrt{v_t}}-\frac{\eta_{t-1}}{\sqrt{v_{t-1}}} ||_1$")
        # plot_curves(opt_dict_dict, "total_clr",
        #             r"$\sum_{t=1}^{T}|| \frac{1}{\sqrt{v_t}}-\frac{1}{\sqrt{v_{t-1}}} ||_1$",
        #             ylim_min=5e-1, ylim_max=1e2)
        # plot_curves(opt_dict_dict, "grad", r"$|g_t|$")
        plot_curves(opt_dict_dict, "fx", r"$f(\theta)$", ylim_min=1e-4, ylim_max=1e2)
        # pickle.dump(opt_dict_dict, open("results/nonconvex_results_average.pkl", 'wb'))
    else:
        b_adam = 0.9

        n_iters = 100000

        adam_lr = 0.8
        ams_lr = 0.8
        madam_lr = 1.2
        print(" lr for Adam: {}, lr for AMSGrad: {}, lr for MAdam: {}".format(adam_lr, ams_lr, madam_lr))
        trials = 100
        sols_adam, sols_amsgrad, sols_madam = [], [], []
        num_succeeds = [0, 0, 0]

        adam_dict_list, madam_dict_list, amsgrad_dict_list = [], [], []
        for tri in range(trials):
            x_adam = 2  # 0
            # x2_adam_max = 0
            x_madam = x_adam
            x_amsgrad = x_adam

            adam_opt = Adam(lr=adam_lr, beta2=b_adam)
            amsgrad_opt = Adam(lr=ams_lr, beta2=b_adam, amsgrad=True)
            madam_opt = MAdam(lr=madam_lr, beta2=1)

            w_madam = 0
            eps = 1e-8

            adam_dict, madam_dict = defaultdict(list), defaultdict(list)
            amsgrad_dict = defaultdict(list)
            for t in range(n_iters):
                sidx = random.randint(1, 11)

                grad_adam = grad_func(x_adam, sidx)
                grad_madam = grad_func(x_madam, sidx)
                grad_amsgrad = grad_func(x_amsgrad, sidx)

                # amsgrad
                x_adam = adam_opt.step(x_adam, grad_adam)
                total_alr, total_astep, total_change_alr = adam_opt.get_metrics()
                update_dict(adam_dict, total_alr, total_astep, total_change_alr, abs(grad_adam), func(x_adam))

                x_amsgrad = amsgrad_opt.step(x_amsgrad, grad_amsgrad)
                total_alr, total_astep, total_change_alr = amsgrad_opt.get_metrics()
                update_dict(amsgrad_dict, total_alr, total_astep, total_change_alr, abs(grad_amsgrad), func(x_amsgrad))

                x_madam = madam_opt.step(x_madam, grad_madam)
                total_alr, total_astep, total_change_alr = madam_opt.get_metrics()
                update_dict(madam_dict, total_alr, total_astep, total_change_alr, abs(grad_madam), func(x_madam))

                sols_adam.append(func(x_adam))
                sols_amsgrad.append(func(x_amsgrad))
                sols_madam.append(func(x_madam))
            adam_dict_list.append(adam_dict)
            madam_dict_list.append(madam_dict)
            amsgrad_dict_list.append(amsgrad_dict)
            if abs(x_adam) < 0.1:
                num_succeeds[0] += 1
            if abs(x_amsgrad) < 0.1:
                num_succeeds[1] += 1
            if abs(x_madam) < 0.1:
                num_succeeds[2] += 1
        print("Adam: {} ({}), AMSGrad: {} ({}), MAdam: {} ({})".format(np.mean(sols_adam), np.std(sols_adam)/math.sqrt(trials),
                                                                       np.mean(sols_amsgrad), np.std(sols_amsgrad) / math.sqrt(trials),
                                                                       np.mean(sols_madam), np.std(sols_madam) / math.sqrt(trials)))
        opt_dict_dict = {"Adam": adam_dict_list, "MAdam": madam_dict_list, "AMSGrad": amsgrad_dict_list}
        plot_curves(opt_dict_dict, "total_alr", r"$\sum_{t=1}^{T} 1/\sqrt{v_t}$")
        plot_curves(opt_dict_dict, "total_astep", r"$\sum_{t=1}^{T}|| g_t/\sqrt{v_t}||^2$")
        plot_curves(opt_dict_dict, "total_clr", r"$\sum_{t=1}^{T}|| \frac{1}{\sqrt{v_t}}-\frac{1}{\sqrt{v_{t-1}}} ||_1$")
        plot_curves(opt_dict_dict, "grad", r"$|g_t|$")
        plot_curves(opt_dict_dict, "fx", r"$f(\theta)$", ylim_min=1e-4, ylim_max=1e2)
        # json.dump(opt_dict_dict, open("results/nonconvex_results_average.json", 'w'))
        pickle.dump(opt_dict_dict, open("results/nonconvex_results_average.pkl", 'wb'))

        print("After optimization: Adam: {},  amsgrad: {}, MADAM: {} num succeeds: {}".format(x_adam, x_madam, x_amsgrad, num_succeeds))
