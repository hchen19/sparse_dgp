from typing import Optional

import math
import scipy.sparse as sp

import torch
from torch import Tensor
from dtmgp.kernels.laplace_kernel import LaplaceProductKernel
from dtmgp.utils.sparse_grid.nsumk import n_sum_k
from dtmgp.utils.operators.torch_scipy_sptransfer import scipy_coo_to_torch_coo, torch_coo_to_scipy_coo

def mk_chol_inv(dyadic_design, 
                markov_kernel = LaplaceProductKernel(), 
                upper: Optional[bool] = True):
    """
    ------------------------
    Parameters:
    dyadic_design: object with dyadic points, dyadic_design.points is [n] size tensor
    markov_kernel: Default = LaplaceProductKernel()
    upper:  flag that indicates whether to return the inverse of a upper or lower triangular matrix
            Default = True
    indices_sort: whether to return the indicies_sort of dyadic_points
            Default = False

    ------------------------
    Returns:
    Rinv: inverse of Cholesky decomposition of markov_kernel(dyadic_points,dyadic_points)
    """
    dyadic_points = dyadic_design.points
    sorted_dyadic, indices_dyadic = dyadic_design.sorted_vals_indices
    indices_sort = dyadic_design.indices_sort
    
    n = len(dyadic_points) # the number of dyadic points
    num_nnz_Rinv = int(3*(n-2) + 2*2) # the number of nonzeros of Rinv

    # initialize sparse_coo indices and values of Rinv
    row_Rinv = torch.zeros(num_nnz_Rinv)
    col_Rinv = torch.zeros(num_nnz_Rinv)
    data_Rinv = torch.zeros(num_nnz_Rinv)
    ii = 0
    
    # indices of closest left neighbors and right neighbors
    left_indices = indices_sort - 1
    right_indices = indices_sort + 1

    for i, dyadic_point in enumerate(dyadic_points):
        # search the closest left neighbor xleft and right neighbor xright
        left_index = left_indices[i]
        right_index = right_indices[i]

        # if dyadic_point is not the boundary point
        if left_index >= 0 and right_index < n:
            xleft = sorted_dyadic[left_index]
            xright = sorted_dyadic[right_index]
            ker_input = torch.tensor([xleft, dyadic_point, xright])
            sys_mat = markov_kernel(x1=ker_input, x2=ker_input) # [3, 3] size tensor
            rhs_vec = torch.tensor([0., 1., 0.])
            coeffs = torch.linalg.solve(sys_mat, rhs_vec) # [3] size tensor
            coeffs = coeffs.div(abs(coeffs[1]).sqrt()) # normalize w.r.t. kernel matrix

            # row_index_xleft = int( (indices_sort == left_index).nonzero(as_tuple=True)[0] )
            # row_index_xright = int( (indices_sort == right_index).nonzero(as_tuple=True)[0] )
            row_index_xleft = indices_sort.tolist().index(left_index)
            row_index_xright = indices_sort.tolist().index(right_index)
            row_Rinv[ii : ii+3] = torch.tensor([row_index_xleft, i, row_index_xright], dtype=int)
            col_Rinv[ii : ii+3] = torch.ones(3, dtype=int)*i
            data_Rinv[ii : ii+3] = coeffs

            ii += 3
        
        # if dyadic_point one point (doesn't have neighbors)
        elif left_index < 0 and right_index >= n:
            # here row_Rinv = tensor([0.]) and col_Rinv = tensor([0.])
            data_Rinv[ii] = 1.
        
        # if dyadic_point is the leftmost point (left_index < 0) or the rightmost point (right_index >= n)
        else:
            if left_index < 0:
                bound_index = right_index
            if right_index >=n:
                bound_index = left_index
            xbound = sorted_dyadic[bound_index]
            ker_input = torch.tensor([xbound, dyadic_point])
            sys_mat = markov_kernel(x1=ker_input, x2=ker_input) # [2, 2] size tensor
            rhs_vec = torch.tensor([0., 1.])
            coeffs = torch.linalg.solve(sys_mat, rhs_vec) # [2] size tensor
            coeffs = coeffs.div(abs(coeffs[1]).sqrt()) # normalize w.r.t. kernel matrix

            row_index_xbound = indices_sort.tolist().index(bound_index)
            row_Rinv[ii : ii+2] = torch.tensor([row_index_xbound, i], dtype=int)
            col_Rinv[ii : ii+2] = torch.ones(2, dtype=int)*i
            data_Rinv[ii : ii+2] = coeffs

            ii += 2
    
    
    Rinv = torch.sparse_coo_tensor(indices=torch.vstack((row_Rinv, col_Rinv)), 
                                   values=data_Rinv, 
                                  size=(n, n))
    
    res = Rinv if upper else Rinv.T
    return res



def tmk_chol_inv(sparse_grid_design, 
                 tensor_markov_kernel=LaplaceProductKernel(),
                 upper: Optional[bool] = True) -> Tensor:
    """
    ------------------------
    Parameters:
    sparse_grid_design: an object with sparse grid design
    tensor_markov_kernel: Default = LaplaceProductKernel()

    upper:  flag that indicates whether to return the inverse of a upper or lower triangular matrix
            Default = True

    ------------------------
    Returns:
    Rinv: inverse of Cholesky decomposition of tensor_markov_kernel(sg_points,sg_points)
    """

    d = sparse_grid_design.d # dimension
    eta = sparse_grid_design.eta # level
    n_sg = sparse_grid_design.n_pts # the size of sparse grids

    indices_set = sparse_grid_design.indices_set # [n_sg, d] size tensor
    list_full = [tuple(l) for l in indices_set.tolist()] # [n_sg, d] size list of tuple
    dict_full = dict((value, idx) for idx,value in enumerate(list_full))

    shape_sg = tuple( [n_sg,]*2 )
    indices_init = torch.empty([2, 0])
    Rinv = torch.sparse_coo_tensor(indices=indices_init, values=[], size=shape_sg).coalesce()

    t_sum_start = max(d, eta-d+1)
    for t_sum in range(t_sum_start, eta+1):
        
        t_arrows = n_sum_k(d,t_sum) # [n_prt, d] size tensor
        n_prt = t_arrows.shape[0]

        for prt in range(n_prt): # loop over partitions of eta(differnet t_arrow for the same eta)
            design_str_prt = sparse_grid_design.design_str_prt[t_sum, prt]# d-dimensional list
            
            # compute Rinv on the full grid points u_fg
            Rinv_fg_scipy_sparse = torch.eye(1).detach().numpy()
            for dim in range(d): # loop over dimension d
                design_str = design_str_prt[dim]
                Rinv_u_torch_sparse = mk_chol_inv(dyadic_design=design_str, 
                                                  markov_kernel=tensor_markov_kernel, 
                                                  upper=upper)
                Rinv_u_scipy_sparse = torch_coo_to_scipy_coo(Rinv_u_torch_sparse)
                Rinv_fg_scipy_sparse = sp.kron(Rinv_fg_scipy_sparse, Rinv_u_scipy_sparse, format="coo")
            #Rinv_sg = scipy_coo_to_torch_coo(Rinv_fg_scipy_sparse)
            
            # get indices and vals of Rinv_fg
            row_Rinv_fg = torch.tensor(Rinv_fg_scipy_sparse.row)
            col_Rinv_fg = torch.tensor(Rinv_fg_scipy_sparse.col)
            indices_Rinv_fg = torch.vstack((row_Rinv_fg, col_Rinv_fg)) # [2, nnz_fg] size tensor
            vals_Rinv_fg = torch.tensor(Rinv_fg_scipy_sparse.data, dtype=float) # [nnz_fg] size tensor

            # get indices in [n_arrow, d] format for [t_sum, prt]-th loop
            indices_select = sparse_grid_design.indices_prt_set[t_sum, prt]
            list_select = [tuple(l) for l in indices_select.tolist()] # [n_arrow, d] size list of tuple

            # get the index in the whole set in [n_arrow] format
            index_select = torch.tensor( [dict_full[x] for x in list_select] ) # [n_arrow] size tensor
            
            # expand Rinv_fg form [n_arrow, n_arrow] sparse matrix to [n_sg, n_sg] sparse matrix
            indices_arrow = index_select[indices_Rinv_fg] # [2, nnz] size tensor
            data_Rinv_arrow = vals_Rinv_fg # [nnz] size tensor
            Rinv_arrow = torch.sparse_coo_tensor(indices=indices_arrow, 
                                                 values=data_Rinv_arrow, 
                                                 size=shape_sg) # [n_sg, n_sg] size tensor
            
            coeff_smolyak = (-1)**(eta-t_sum) * math.comb(d-1, eta-t_sum) # scalar
            Rinv += coeff_smolyak * Rinv_arrow

    res = Rinv if upper else Rinv.T
    return res