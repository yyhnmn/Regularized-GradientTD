import numpy as np

colorList={
    0:'b',
    1:'g',
    2:'r',
    3:'c',
    4:'y',
    5:'k',
    6:'m'
    }
def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def get_random_colour():
    return (np.random.rand(3, ) * 0.2) + 0.4

def plot(ax, data, index,label=None, color = None,):
    if color is None:
        color = colorList[index]
    mean, ste, runs = data
    base, = ax.plot(mean, label=label, color=color, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.1)
