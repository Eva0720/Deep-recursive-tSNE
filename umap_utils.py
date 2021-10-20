# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:10:39 2020

@author: Summer
"""

from __future__ import print_function



import locale

from warnings import warn

import time



from scipy.optimize import curve_fit

from sklearn.base import BaseEstimator

from sklearn.utils import check_random_state, check_array

from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import normalize

from sklearn.neighbors import KDTree



try:

    import joblib

except ImportError:

    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23

    from sklearn.externals import joblib



import numpy as np

import scipy.sparse

import scipy.sparse.csgraph

import numba



import distances as dist



import sparse as sparse

import sparse_nndescent as sparse_nn



from utils import (

    tau_rand_int,

    deheap_sort,

    submatrix,

    ts,

    csr_unique,

    fast_knn_indices,

)

from rp_tree import rptree_leaf_array, make_forest

from nndescent import (

    # make_nn_descent,

    # make_initialisations,

    # make_initialized_nnd_search,

    nn_descent,

    initialized_nnd_search,

    initialise_search,

)

from rp_tree import rptree_leaf_array, make_forest

from spectral import spectral_layout

from utils import deheap_sort, submatrix

from layouts import (

    optimize_layout_euclidean,

    optimize_layout_generic,

    optimize_layout_inverse,

)



try:

    # Use pynndescent, if installed (python 3 only)

    from pynndescent import NNDescent

    from pynndescent.distances import named_distances as pynn_named_distances

    from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances



    _HAVE_PYNNDESCENT = True

except ImportError:

    _HAVE_PYNNDESCENT = False



locale.setlocale(locale.LC_NUMERIC, "C")





#def Hbeta(D, beta):
#    P = np.exp(-D * beta)
#    sumP = np.sum(P)
#    H = np.log(sumP) + beta * np.sum(D * P) / sumP
#    P = P / sumP
#    return H, P
#
#def x2p_job(data):
#    i, Di, tol, logU = data
#    beta = 1.0
#    betamin = -np.inf
#    betamax = np.inf
#    H, thisP = Hbeta(Di, beta)
#
#    Hdiff = H - logU
#    tries = 0
#    while np.abs(Hdiff) > tol and tries < 50:
#        if Hdiff > 0:
#            betamin = beta
#            if betamax == -np.inf:
#                beta = beta * 2
#            else:
#                beta = (betamin + betamax) / 2
#        else:
#            betamax = beta
#            if betamin == -np.inf:
#                beta = beta / 2
#            else:
#                beta = (betamin + betamax) / 2
#
#        H, thisP = Hbeta(Di, beta)
#        Hdiff = H - logU
#        tries += 1
#
#    return i, thisP
#
#
#def x2p(X,perplexity):
#    tol = 1e-5
#    n = X.shape[0]
#    logU = np.log(perplexity)
#  
#    sum_X = np.sum(np.square(X), axis=1)
#    D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))
#
#    idx = (1 - np.eye(n)).astype(bool)
#    D = D[idx].reshape([n, -1])
#
#    result=[]
#    for i in range(n):
#        data_setin=i, D[i], tol, logU
#        result1=x2p_job(data_setin)
#        result.append(result1)
#    P = np.zeros([n, n])
#    for i, thisP in result:
#        P[i, idx[i]] = thisP
#    return P



def hd_v(X):
    
        dmat = dist.pairwise_special_metric(X, metric="euclidean", kwds=None)
        random_state = 71
        random_state = check_random_state(random_state)
        _n_neighbors = 15
        #_n_neighbors = 15
        _metric_kwds={}
        angular_rp_forest=False
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        verbose=False
        graph_, _sigmas, _rhos = fuzzy_simplicial_set(

                dmat,

                _n_neighbors,

                random_state,

                "euclidean",

                _metric_kwds,

                None,

                None,

                angular_rp_forest,

                set_op_mix_ratio,

                local_connectivity,

                True,

                verbose,

            )
 #       hr=spectral_layout(X, graph_, dim, random_state, "euclidean", _metric_kwds)
        return graph_
        


INT32_MIN = np.iinfo(np.int32).min + 1

INT32_MAX = np.iinfo(np.int32).max - 1



SMOOTH_K_TOLERANCE = 1e-5

MIN_K_DIST_SCALE = 1e-3

NPY_INFINITY = np.inf



def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    
    target = np.log2(k) * bandwidth

    rho = np.zeros(distances.shape[0], dtype=np.float32)

    result = np.zeros(distances.shape[0], dtype=np.float32)



    mean_distances = np.mean(distances)



    for i in range(distances.shape[0]):

        lo = 0.0

        hi = NPY_INFINITY

        mid = 1.0



        # TODO: This is very inefficient, but will do for now. FIXME

        ith_distances = distances[i]

        non_zero_dists = ith_distances[ith_distances > 0.0]

        if non_zero_dists.shape[0] >= local_connectivity:

            index = int(np.floor(local_connectivity))

            interpolation = local_connectivity - index

            if index > 0:

                rho[i] = non_zero_dists[index - 1]

                if interpolation > SMOOTH_K_TOLERANCE:

                    rho[i] += interpolation * (

                        non_zero_dists[index] - non_zero_dists[index - 1]

                    )

            else:

                rho[i] = interpolation * non_zero_dists[0]

        elif non_zero_dists.shape[0] > 0:

            rho[i] = np.max(non_zero_dists)



        for n in range(n_iter):



            psum = 0.0

            for j in range(1, distances.shape[1]):

                d = distances[i, j] - rho[i]

                if d > 0:

                    psum += np.exp(-(d / mid))

                else:

                    psum += 1.0



            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:

                break



            if psum > target:

                hi = mid

                mid = (lo + hi) / 2.0

            else:

                lo = mid

                if hi == NPY_INFINITY:

                    mid *= 2

                else:

                    mid = (lo + hi) / 2.0



        result[i] = mid



        # TODO: This is very inefficient, but will do for now. FIXME

        if rho[i] > 0.0:

            mean_ith_distances = np.mean(ith_distances)

            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:

                result[i] = MIN_K_DIST_SCALE * mean_ith_distances

        else:

            if result[i] < MIN_K_DIST_SCALE * mean_distances:

                result[i] = MIN_K_DIST_SCALE * mean_distances



    return result, rho



def nearest_neighbors(

    X,

    n_neighbors,

    metric,

    metric_kwds,

    angular,

    random_state,

    low_memory=False,

    use_pynndescent=True,

    verbose=False,

):



    if verbose:

        print(ts(), "Finding Nearest Neighbors")



    if metric == "precomputed":

        # Note that this does not support sparse distance matrices yet ...

        # Compute indices of n nearest neighbors

        knn_indices = fast_knn_indices(X, n_neighbors)

        # knn_indices = np.argsort(X)[:, :n_neighbors]

        # Compute the nearest neighbor distances

        #   (equivalent to np.sort(X)[:,:n_neighbors])

        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()



        rp_forest = []

    else:

        # TODO: Hacked values for now

        n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))

        n_iters = max(5, int(round(np.log2(X.shape[0]))))



        if _HAVE_PYNNDESCENT and use_pynndescent:

            nnd = NNDescent(

                X,

                n_neighbors=n_neighbors,

                metric=metric,

                metric_kwds=metric_kwds,

                random_state=random_state,

                n_trees=n_trees,

                n_iters=n_iters,

                max_candidates=60,

                low_memory=low_memory,

                verbose=verbose,

            )

            knn_indices, knn_dists = nnd.neighbor_graph

            rp_forest = nnd

        else:

            # Otherwise fall back to nn descent in umap

            if callable(metric):

                _distance_func = metric

            elif metric in dist.named_distances:

                _distance_func = dist.named_distances[metric]

            else:

                raise ValueError("Metric is neither callable, nor a recognised string")



            rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)



            if scipy.sparse.isspmatrix_csr(X):

                if callable(metric):

                    _distance_func = metric

                else:

                    try:

                        _distance_func = sparse.sparse_named_distances[metric]

                        if metric in sparse.sparse_need_n_features:

                            metric_kwds["n_features"] = X.shape[1]

                    except KeyError as e:

                        raise ValueError(

                            "Metric {} not supported for sparse data".format(metric)

                        ) from e



                # Create a partial function for distances with arguments

                if len(metric_kwds) > 0:

                    dist_args = tuple(metric_kwds.values())



                    @numba.njit()

                    def _partial_dist_func(ind1, data1, ind2, data2):

                        return _distance_func(ind1, data1, ind2, data2, *dist_args)



                    distance_func = _partial_dist_func

                else:

                    distance_func = _distance_func

                # metric_nn_descent = sparse.make_sparse_nn_descent(

                #     distance_func, tuple(metric_kwds.values())

                # )



                if verbose:

                    print(ts(), "Building RP forest with", str(n_trees), "trees")



                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)

                leaf_array = rptree_leaf_array(rp_forest)



                if verbose:

                    print(ts(), "NN descent for", str(n_iters), "iterations")

                knn_indices, knn_dists = sparse_nn.sparse_nn_descent(

                    X.indices,

                    X.indptr,

                    X.data,

                    X.shape[0],

                    n_neighbors,

                    rng_state,

                    max_candidates=60,

                    sparse_dist=distance_func,

                    low_memory=low_memory,

                    rp_tree_init=True,

                    leaf_array=leaf_array,

                    n_iters=n_iters,

                    verbose=verbose,

                )

            else:

                # metric_nn_descent = make_nn_descent(

                #     distance_func, tuple(metric_kwds.values())

                # )

                if len(metric_kwds) > 0:

                    dist_args = tuple(metric_kwds.values())



                    @numba.njit()

                    def _partial_dist_func(x, y):

                        return _distance_func(x, y, *dist_args)



                    distance_func = _partial_dist_func

                else:

                    distance_func = _distance_func



                if verbose:

                    print(ts(), "Building RP forest with", str(n_trees), "trees")

                rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)

                leaf_array = rptree_leaf_array(rp_forest)

                if verbose:

                    print(ts(), "NN descent for", str(n_iters), "iterations")

                knn_indices, knn_dists = nn_descent(

                    X,

                    n_neighbors,

                    rng_state,

                    max_candidates=60,

                    dist=distance_func,

                    low_memory=low_memory,

                    rp_tree_init=True,

                    leaf_array=leaf_array,

                    n_iters=n_iters,

                    verbose=False,

                )



            if np.any(knn_indices < 0):

                warn(

                    "Failed to correctly find n_neighbors for some samples."

                    "Results may be less than ideal. Try re-running with"

                    "different parameters."

                )

    if verbose:

        print(ts(), "Finished Nearest Neighbor Search")

    return knn_indices, knn_dists, rp_forest





@numba.njit(

    locals={

        "knn_dists": numba.types.float32[:, ::1],

        "sigmas": numba.types.float32[::1],

        "rhos": numba.types.float32[::1],

        "val": numba.types.float32,

    },

    parallel=True,

    fastmath=True,

)

def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos):


    n_samples = knn_indices.shape[0]

    n_neighbors = knn_indices.shape[1]



    rows = np.zeros(knn_indices.size, dtype=np.int32)

    cols = np.zeros(knn_indices.size, dtype=np.int32)

    vals = np.zeros(knn_indices.size, dtype=np.float32)



    for i in range(n_samples):

        for j in range(n_neighbors):

            if knn_indices[i, j] == -1:

                continue  # We didn't get the full knn for i

            if knn_indices[i, j] == i:

                val = 0.0

            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:

                val = 1.0

            else:

                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))



            rows[i * n_neighbors + j] = i

            cols[i * n_neighbors + j] = knn_indices[i, j]

            vals[i * n_neighbors + j] = val



    return rows, cols, vals




def fuzzy_simplicial_set(

    X,

    n_neighbors,

    random_state,

    metric,

    metric_kwds={},

    knn_indices=None,

    knn_dists=None,

    angular=False,

    set_op_mix_ratio=1.0,

    local_connectivity=1.0,

    apply_set_operations=True,

    verbose=False,

):

    if knn_indices is None or knn_dists is None:

        knn_indices, knn_dists, _ = nearest_neighbors(

            X, n_neighbors, metric, metric_kwds, angular, random_state, verbose=verbose

        )



    knn_dists = knn_dists.astype(np.float32)



#    sigmas, rhos = smooth_knn_dist(
#
#        knn_dists, float(n_neighbors), local_connectivity=float(local_connectivity),
#
#    )
    sigmas, rhos = smooth_knn_dist(

        knn_dists, float(n_neighbors), local_connectivity=1.0,

    )


    rows, cols, vals = compute_membership_strengths(

        knn_indices, knn_dists, sigmas, rhos

    )



    result = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(X.shape[0], X.shape[0]))
   # print(result)
    transpose = result.transpose()



    prod_matrix = result.multiply(transpose)
  #  set_op_mix_ratio=1.0
    
    result = result + transpose - prod_matrix




#    result = (
#
#            set_op_mix_ratio * (result + transpose - prod_matrix)
#
#            + (1.0 - set_op_mix_ratio) * prod_matrix
#
#        )

#    result.eliminate_zeros()
#
#
#
#    if apply_set_operations:
#
#        transpose = result.transpose()
#
#
#
#        prod_matrix = result.multiply(transpose)
#        set_op_mix_ratio=1.0
#
#
#        result = (
#
#            set_op_mix_ratio * (result + transpose - prod_matrix)
#
#            + (1.0 - set_op_mix_ratio) * prod_matrix
#
#        )
#
#
#
#    result.eliminate_zeros()



    return result, sigmas, rhos






