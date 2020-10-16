import pickle
from matplotlib import pyplot as plt
import matplotlib
from collections import defaultdict
import numpy as np
font = { 'size'   : 16}

matplotlib.rc('font', **font)

fpath = "quadratic_trajs/res_hs_1_20_sigma_0.2_0.2.pkl"
traj_dict = pickle.load(open(fpath, 'rb'))

def parse_dict(err_dict):
    best_err_dict = defaultdict(lambda : (float("inf"), 0))
    for name, errs in err_dict.items():
        lr = float(name.split('_')[0][2:])
        mean, std = np.mean(errs), np.std(errs)/np.sqrt(len(errs))
        if mean < best_err_dict[lr][0]:
            best_err_dict[lr] = (mean, std)

    lr_list = list(best_err_dict.keys())
    err_list = np.array([best_err_dict[lr][0] for lr in lr_list])
    std_list = np.array([best_err_dict[lr][1] for lr in lr_list])

    return lr_list, err_list, std_list

adam_lrs, adam_errs, adam_stds = parse_dict(traj_dict['AdamErrs'])
madam_lrs, madam_errs, madam_stds = parse_dict(traj_dict['MadamErrs'])

def plot_errs(lr_list, err_list, std_list, c1="or", c2="gray", label="MAdam"):
    plt.plot(lr_list, err_list, c1, label=label)
    plt.plot(lr_list, err_list, '-', color=c2)

    plt.fill_between(lr_list, err_list - std_list, err_list + std_list,
                     color=c2, alpha=0.2)

plot_errs(adam_lrs, adam_errs, adam_stds, c1="^y", c2="lightskyblue", label="Adam")
plot_errs(madam_lrs, madam_errs, madam_stds, c1="or", c2="gray", label="MAdam")

plt.title("Errors on NQM (1000 steps)")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Error")
plt.yscale("log")
plt.legend()
plt.tight_layout()

plt.savefig("compelling_example.pdf")
# plt.show()
