# import standard scientific packages
import numpy as np
from scipy.sparse import (csr_matrix, spdiags, eye)
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.validation import check_array
from manilearn.utils.neighbors import KnnSolver


# compute adjacency matrix
def adjacency(X, n_neighbors=10, radius=1.5, algorithm='brute', method='radius',
              weight='connectivity', nearest_neighbors_kwargs=None,
              adjacency_kwargs=None):
    """Computes an adjacency matrix

    Parameters
    ----------
    X : array, ( N x D )
        data matrix

    n_neighbors : int, default = 5
        number of nearest neighbors

    radius : int, default = 1.5
        radius to find the nearest neighbors

    method : str, default = 'radius'
        ['radius' | 'knn' ]
        the type of nearest neighbors algorithm

    weight : str, default = 'connectivity'
        ['connectivity'|'heat'|'angle']
        the scaling kernel function to use on the distance values found from
        the neighbors algorithm

    algorithm : str, default = 'brute'
        ['annoy'|'brute'|'kd_tree'|'ball_tree'|'hdidx'|'pyflann'|'cyflann']
        algorithm to find the k-nearest or radius-nearest neighbors

    nearest_neighbors_kwargs : dict, default = None
        a dictionary of key values for the KnnSolver class

    adjacency_kwargs : dict, default = None
        a dictionary of key values for the adjacency matrix construction

    Returns
    -------
    adjacency_matrix : array, ( N x N )
        a weighted adjacency matrix

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com

    References
    ----------
    TODO: code references
    TODO: paper references
    """

    if algorithm in ['annoy'] and method in ['radius']:
        method = 'knn'

    # initialize the knn model with available parameters
    knn_model = KnnSolver(n_neighbors=n_neighbors,
                          radius=radius,
                          algorithm=algorithm,
                          method=method,
                          algorithm_kwargs=nearest_neighbors_kwargs)

    # find the nearest neighbors indices and distances
    distances, indices = knn_model.find_knn(X)

    # construct adjacency matrix
    if adjacency_kwargs is None:
        adjacency_kwargs = {}

    adjacency_matrix = create_adjacency(distances, indices, X, method=method,
                                        weight=weight,
                                        **adjacency_kwargs)

    return adjacency_matrix


def create_adjacency(distances, indices, data, method='radius', mode='distance',
                     weight='connectivity', gamma=1.0):
    """This function will create a sparse symmetric
    weighted adjacency matrix from nearest neighbors
    and their corresponding distances.

    Parameters:
    -----------
    indices : array, (N x k)
        an Nxk array where M are the number of data points and k
        are the k-1 nearest neighbors connected to that data point M.

    distances : array, (N x k)
        an MxN array where N are the number of data points and k are
        the k-1 nearest neighbor distances connected to that data point M.

    data : array, (N x D)
        an NxD array where N is the number of data points and D is the number
        of dimensions.

    method : str, default = 'knn'
        ['knn' | 'radius']
        base algorithm to find the nearest neighbors

    weight : str, default = 'heat'
        ['heat' | 'angle' | 'connectivity']
        weights to put on the data points

    gamma : float, default 1.0
        the spread for the weight function

    Returns:
    --------
    Adjacency Matrix          - a sparse MxM sparse weighted adjacency
                                matrix.

    References
    ----------
    Uses code from the sklearn library, specifically the neighbors, base
    class with functions k_neighbors_graph and radius_neighbors_graph
        https://goo.gl/DKtpBX
    """
    # Separate, tile and ravel the neighbours from their
    # corresponding points

    # check array (sparse is allowed)
    data = check_array(data, accept_sparse=['csr', 'csc', 'coo'])

    # dimensions for samples
    n_samples2 = data.shape[0]

    if method in ['radius']:

        # construct CSR matrix representation of the NN graph
        if weight in ['connectivity']:
            A_ind = indices
            A_data = None

        elif weight not in ['distance']:
            A_ind = indices
            A_data = np.concatenate(list(distances))

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                'or "distance" but got %s instead' % mode)

        n_samples1 = A_ind.shape[0]
        n_neighbors = np.array([len(a) for a in A_ind])
        A_ind = np.concatenate(list(A_ind))
        if A_data is None:
            A_data = np.ones(len(A_ind))
        A_indptr = np.concatenate((np.zeros(1, dtype=int),
                                   np.cumsum(n_neighbors)))

    elif method in ['knn']:

        n_samples1 = data.shape[0]

        n_neighbors = distances.shape[1]

        n_nonzero = n_samples1 * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        # construct CSR matrix representation of the k-NN graph
        if weight in ['connectivity']:
            A_data = np.ones(n_samples1 * n_neighbors)
            A_ind = indices

        elif weight not in ['distance']:
            A_data, A_ind = distances, indices
            A_data = np.ravel(A_data)

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity" '
                'or "distance" but got "%s" instead' % mode)

    else:
        raise ValueError(
            'Unrecognized method of graph construction. Must be '
            '"knn" or "radius" but got "{alg}" instead'.format(alg=method))

    # compute weights
    if weight in ['connectivity']:
        pass

    elif weight in ['heat']:
        A_data = np.exp(-A_data**2 / gamma**2)

    elif weight in ['angle']:
        A_data = np.exp(-np.arccos(1-A_data))

    else:
        raise ValueError('Sorry. Unrecognized affinity weight.')

    # Create the sparse matrix
    if method in ['knn']:
        adjacency_mat = csr_matrix((A_data, A_ind.ravel(), A_indptr),
                                   shape=(n_samples1, n_samples2))

    elif method in ['radius']:
        adjacency_mat = csr_matrix((A_data, A_ind, A_indptr),
                                   shape=(n_samples1, n_samples2))

    # Make sure the matrix is symmetric
    adjacency_mat = maximum(adjacency_mat, adjacency_mat.T)

    return adjacency_mat


def create_constraint(adjacency_matrix, constraint='degree'):
    """Computes the constraint matrix from a weighted adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : dense, sparse (N x N)
        weighted adjacency matrix.

    constraint : str, default = 'degree'
        ['identity'|'degree'|'similarity'|'dissimilarity']
        the type of constraint matrix to construct

    Returns
    -------
    D : array, sparse (N x N)
        constraint matrix
    """
    if constraint in ['degree']:
        D = spdiags(data=np.squeeze(np.asarray(adjacency_matrix.sum(axis=1))),
                    diags=[0], m=adjacency_matrix.shape[0],
                    n=adjacency_matrix.shape[0])

        return D
    elif constraint in ['identity']:
        D = eye(m=adjacency_matrix.shape[0], n=adjacency_matrix.shape[0],k=0)

        return D
    else:
        raise NotImplementedError('No other methods implemented.')


def create_laplacian(adjacency_matrix, laplacian='unnormalized'):
    """Computes the graph laplacian from a weighted adjacency matrix.

    Parameters
    ----------
    adjacency_matrix : dense, sparse (N x N)
        weighted adjacency matrix.

    laplacian : str, default = 'unnormalized'
        ['normalized'|'unnormalized'|'randomwalk']
        the type of laplacian matrix to construct

    Returns
    -------
    L : sparse (N x N)
        Laplacian matrix.

    D : sparse (N x N)
        Diagonal degree matrix.

    References
    ----------
    sklearn - SpectralEmbedding
    megaman - spectral_embedding

    TODO: implement random walk laplacian
    TODO: implement renormalized laplacian
    """
    if laplacian in ['unnormalized']:
        laplacian_matrix, diagonal_matrix = \
            graph_laplacian(adjacency_matrix, normed=False, return_diag=True)

        diagonal_matrix = spdiags(data=diagonal_matrix, diags=[0],
                                  m=adjacency_matrix.shape[0],
                                  n=adjacency_matrix.shape[0])

        return laplacian_matrix, diagonal_matrix

    elif laplacian in ['normalized']:
        laplacian_matrix = graph_laplacian(adjacency_matrix,
                                           normed=True,
                                           return_diag=False)

        return laplacian_matrix

    else:
        raise ValueError('Unrecognized Graph Laplacian.')


def maximum(mata, matb):
    """This gives you the element-wise maximum between two sparse
    matrices of size (nxn)

    Reference
    ---------
        http://goo.gl/k0Yfmk
    """
    bisbigger = mata-matb
    bisbigger.data = np.where(bisbigger.data < 0, 1, 0)
    return mata - mata.multiply(bisbigger) + matb.multiply(bisbigger)

