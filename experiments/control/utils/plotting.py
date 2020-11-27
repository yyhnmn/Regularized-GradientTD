import numpy as np

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def get_random_colour():
    return (np.random.rand(3, ) * 0.20) + 0.40

def plot(ax, data, label=None, color = None):
    if color is None:
        color = get_random_colour()
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, color=color, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)