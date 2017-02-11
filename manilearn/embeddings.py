"""
Date Created: 5th February, 2017

Author  : J. Emmanuel Johnson
Date    : 5th February, 2017
Email   : emanjohnson91@gmail.com
"""
import numpy as np
from manilearn.utils.graph import (create_laplacian, create_constraint)
# from manilearn.schroedinger import potential_tradeoff
from megaman.utils.eigendecomp import eigen_decomposition
# TODO: default eigensolver_kwargs
# TODO: default regularize_kwargs


def graph_embedding(adjacency_matrix, n_components=2, operator='ssse',
                    constraint='degree', regularize_mat=None,
                    eigen_solver='auto', eigensolver_kwargs=None,
                    regularize_kwargs=None):
    """Graph Embedding algorithm.

    This algorithm computes the graph embedding using an adjacency matrix and
    multiple regularization options. It solves the problem using an eigenvalue
    decomposition method to produce an embedding (subspace).

    Parameters
    ----------
    adjacency_matrix : array, ( N x N )
        a weighted adjacency matrix to perform the graph embedding

    n_components : integer, default = 2
        number of coordinates for the manifold embedding

    operator : string, default = 'lap'
        ['lap'|'normlap'|'randomwalk'|'ssse'|'sepl' | 'sssepl' | 'ssseplp']
        the Schroedinger operator to use for the Schroedinger Eigenmaps algorithm

    constraint : str, default = 'degree'
        ['degree'|'identity'|'similarity'|'dissimilarity]
        the constraint matrix when solving the eigenvalue decomposition problem

    eigen_solver : string, default = 'lobpcg'
        ['auto' | 'dense' | 'arpack' | 'lobpcg'|'amg']
        eigenvalue decomposition solver

    eigensolver_kwargs : dict,
        a dictionary of key word arguments for the eigensolver

    regularize_mat : array, ( N x N )
        a regularization matrix used as an augmentation to the problem

    regularize_kwargs : dict
        a dictionary of key word arguments to

    Returns
    -------
    eigvals : array, ( 1 x n_components)
        eigenvalues of the embedding
    eigevecs : array, ( N x n_components)
        eigenvectors of the embedding

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com

    References
    ----------
    TODO: deng cai, cahill, niyogi
    TODO: similarity, dissimilarity constraint matrix
    """

    # create laplacian and diagonal degree matrix
    if operator in ['normlap']:
        laplacian = 'normalized'
    else:
        laplacian = 'unnormalized'

    lap_mat, constraint_mat = create_laplacian(adjacency_matrix,
                                               laplacian=laplacian)

    # create normalization matrix
    constraint_mat = create_constraint(adjacency_matrix,
                                       constraint=constraint)

    # Laplacian operators
    if operator in ['lap', 'normlap', 'randomwalk']:

        A_mat = lap_mat         # laplacian matrix
        # TODO: implement normalized Laplacian
        # TODO: implement random-walk Laplacian

    # Schroedinger operators
    elif operator in ['ssse', 'sepl', 'sssepl', 'sssepl']:

        if regularize_kwargs is None or 'alpha' not in regularize_kwargs.keys(): # or regularize_kwargs['alpha'] is None:
            alpha = potential_tradeoff(lap_mat, regularize_mat)
        else:
            alpha = regularize_kwargs['alpha']
            alpha = potential_tradeoff(lap_mat, regularize_mat, weight=alpha)

        A_mat = lap_mat + alpha * regularize_mat

    else:
        raise ValueError('Unrecognized operator.')

    # solve the eigenvalue decomposition problem

    if eigen_solver in ['arpack']:
        if eigensolver_kwargs is None:
            eigensolver_kwargs = {'M': constraint_mat}
        else:
            eigensolver_kwargs['M'] = constraint_mat

    elif eigen_solver in ['dense']:

        # transform matrices to dense arrays
        A_mat = A_mat.toarray()
        constraint_mat = constraint_mat.toarray()

        if eigensolver_kwargs is None:
            eigensolver_kwargs = {'b': constraint_mat}
        else:
            eigensolver_kwargs['b'] = constraint_mat

    elif eigen_solver in ['lobpcg', 'pyamg']:
        if eigensolver_kwargs is None:
            eigensolver_kwargs = {'B': constraint_mat}
        else:
            eigensolver_kwargs['B'] = constraint_mat

    else:
        raise ValueError('Unrecognized eigensolver')

    eigenvalues, embedding = eigen_decomposition(A_mat, n_components=n_components,
                                                 eigen_solver=eigen_solver,
                                                 solver_kwds=eigensolver_kwargs)
    # eig_model = EigSolver(n_components=n_components, eigsolver=eigen_solver,
    #                       eigensolver_kwargs=eigensolver_kwargs)
    #
    # eigvals, eigvecs = eig_model.find_eig(A=A_mat, B=B_mat)

    return eigenvalues, embedding


# TODO: fix import error with potential trade off
# Determine appropriate trade off parameter between L and D
def potential_tradeoff(L, V, weight=17.78):
    """Gives the suggested value of alpha:

             trace of potential
    weight * ------------------ = suggested weight
             trace of Laplacian

    Parameters
    ----------
    L : array, sparse ( N x N )
        sparse Laplacian matrix

    V : array, sparse ( N x N )
        sparse Schroedinger potential matrix

    weight : float, default = 17.78
        trade-off parameter between the Laplacian matrix and the Schroedinger
        potential matrix

    Returns
    -------
    new_weight : float
        scaled weight parameter to trade off the Laplacian and the Potential matrix

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 4th February
    Email   : emanjohnson91@gmail.com
    """
    return weight * (np.trace(L.todense()) / np.trace(V.todense()))

