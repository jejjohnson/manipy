import matplotlib.pyplot as plt
from time import time
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import SpectralEmbedding
from manilearn.schroedinger import SchroedingerEigenmaps


# swiss roll test to test out my function versus theirs
def swiss_roll_test():

    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    n_neighbors = 20
    n_components = 2

    # original lE algorithm
    t0 = time()
    ml_model = SpectralEmbedding(n_neighbors=n_neighbors,
                                 n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='scikit')
    ax[0].set_title('Sklearn-LE: {t:.2g} secs'.format(t=t1-t0))


    # Schroedinger Eigenmaps
    t0 = time()
    potential_kwargs = {'image': X}

    ml_model = SchroedingerEigenmaps(n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     eigen_solver='dense',
                                     alpha=100,
                                     potential_kwargs=potential_kwargs)
    Y = ml_model.fit_transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='Schroedinger')
    ax[1].set_title('SSSE: {t:.2g}'.format(t=t1-t0))

    plt.show()

# TODO: Digits embedding test
# TODO: HSI classification test


if __name__ == "__main__":

     swiss_roll_test()
     # hsi_test()
