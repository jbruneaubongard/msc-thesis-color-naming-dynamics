import pickle
from src.ib_model import *
from src.encoders import *
import matplotlib.pyplot as plt
import numpy as np

def generate_color_gradient(num_colors):
    gradient = np.linspace(0.5, 1, num_colors)
    colors = plt.cm.Purples(gradient)
    return colors

plt.style.use('../src/_.scientific.mplstyle')

model = load_model()

def plot_trajectory(dataset, mode, ix, ax=None, save=False):
    file = f'../results/trajectory_{dataset}_{mode}_{ix}.pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)
        initial_encoder = data[-2]
        data = data[-1]
    colors = generate_color_gradient(len(data))

    if ax is None:
        _, ax = plt.subplots(figsize=(6,4))

    ax.plot(model.IB_curve[0], model.IB_curve[1], label='IB curve', color='teal')
    ax.scatter(model.complexity(initial_encoder), model.accuracy(initial_encoder), 
               color='r', s=10, label='Initial Language', zorder=3)
    
    for i,enc in enumerate(data):
        ax.scatter(model.complexity(enc), model.accuracy(enc), color=colors[i], s=10, alpha=0.5, zorder=2)

    plt.colorbar(plt.cm.ScalarMappable(cmap='Purples'), label='Iteration', values = range(len(data)), ax=ax)
    ax.set_xlabel('Complexity')
    ax.set_ylabel(r'Informativity + I(M,U)')
    ax.legend()
    if dataset=='shuffled_voronoi':
        ax.set_title(f'Dataset: sv, Algorithm: {mode}, Index: {ix}')
    else:
        ax.set_title(f'Dataset: {dataset}, Algorithm: {mode}, Index: {ix}')

    if save:
        plt.savefig(f'../figures/trajectory_{dataset}_{mode}_{ix}.pdf', dpi=150)
        plt.close()
    
    

def plot_initial_final_palettes(dataset, mode, ix, save=False):
    file = f'../results/trajectory_{dataset}_{mode}_{ix}.pkl'

    with open(file, 'rb') as f:
        data = pickle.load(f)
        initial_encoder = Encoder(data[-2])
        final_encoder = Encoder(data[0])

    fig, ax = plt.subplots(2, 1, figsize=(10,8))

    initial_encoder.plot_palette(ax=ax[0])
    final_encoder.plot_palette(ax=ax[1])
    
    ax[0].set_title('Initial language')
    ax[1].set_title('Final language')

    if dataset=='shuffled_voronoi':
        fig.suptitle(f'Dataset: sv, Algorithm: {mode}, Index: {ix}', fontsize=16)
    else:
        fig.suptitle(f'Dataset: {dataset}, Algorithm: {mode}, Index: {ix}', fontsize=16)

    if save:
        plt.savefig(f'../figures/palettes_{dataset}_{mode}_{ix}.pdf', dpi=150)
        plt.close()
    
    
    