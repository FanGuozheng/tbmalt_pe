"""A toolkit for feature analysis."""
from typing import Union, List
import numpy as np
from  torch import Tensor
from sklearn.decomposition import PCA as sklCPA
import matplotlib.pyplot as plt

# TMP code
import matplotlib.colors as mcolors
overlap = {name for name in mcolors.CSS4_COLORS
           if f'xkcd:{name}' in mcolors.XKCD_COLORS}



class PCA:

    def __init__(self):
        pass

    @staticmethod
    def scikitlearn(X: Union[np.ndarray, Tensor],
                    y: Union[List[str], np.ndarray],
                    n_components: int = 2,
                    plot: bool = True):

        pca = sklCPA(n_components=n_components)
        X_r = pca.fit(X).transform(X)

        if plot:
            for color, i, target_name in zip(mcolors.CSS4_COLORS.keys(), [0, 1, 2], y):
                plt.scatter(
                    X_r[y == i, 0], X_r[y == i, 1], color=mcolors.CSS4_COLORS[color], alpha=0.8,
                    # label=target_name
                )
            plt.legend(loc="best", shadow=False, scatterpoints=1)
            plt.title("PCA of dataset features")

class TSNE():
    pass
