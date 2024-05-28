import numpy as np 
import matplotlib.pyplot as plt
from .ib_model import load_model
from .tools import *
from .figures import *


# Standard IB model
model = load_model()

np.random.seed(19)

def get_ib_curve():
    return model.IB_curve

class Encoder:

    def __init__(
            self,
            encoding_matrix: np.ndarray,
            pM=model.pM,
            pU_M=model.pU_M,
    ):
        assert np.isclose(np.sum(encoding_matrix, axis=1), 1).all(), "Rows of the encoding matrix do not sum to 1."

        self.encoding_matrix = encoding_matrix
        self.pM = pM
        self.pU_M = pU_M
    
    @property
    def voc_size(self):
        return np.shape(self.encoding_matrix)[1]
    
    @property
    def word_distribution(self):
        return self.encoding_matrix.T @ self.pM

    @property
    def complexity(self):
        return MI(self.encoding_matrix * self.pM)
    
    @property
    def informativity(self):
        pMW = self.encoding_matrix * self.pM
        pWU = pMW.T @ self.pU_M
        return MI(pWU)
    
    @property
    def partition(self):
        return encoder_to_partition(self.encoding_matrix)

    def plot_palette(
        self,
        ax=None,
        save_path=None,
    ):
        
        n = self.encoding_matrix.shape[0]
        joint_proba_meaning_words = self.encoding_matrix * np.ones((n, 1))/n
        grid = get_color_grid(joint_proba_meaning_words)
        img = np.flipud(grid2img(grid))
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        clrs = np.array([r.flatten(), g.flatten(), b.flatten()]).T

        if ax is None:
            _, ax = plt.subplots()

        ax.set_axis_off()
        ax = ax.pcolor(r, color=clrs, linewidth=0.04, edgecolors='None')
        ax.set_array(None)
        plt.xlim([0, 42])
        plt.ylim([0, 10])
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        
        
        return ax
    
    