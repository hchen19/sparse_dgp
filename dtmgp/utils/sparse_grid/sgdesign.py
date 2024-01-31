#!/usr/bin/env python3
import torch
from dtmgp.utils.sparse_grid.nsumk import n_sum_k
from dtmgp.utils.sparse_grid.hyperbolic_cross import hc_design

class sg:
    """
    ------------------------
    Parameters:
    d: dimension of input, d >= 2
    eta: level of sparse grid design
    design_fun: design used in one dimension, default = hc_design (Here we assume the sparse grid design is the same in each dimension)
    input_bd: [d,2] size tensor, default value = [0, 1]^d

    ------------------------
    Returns:
    self: an object with sparse grid design
    """

    def __init__(self, d, eta, input_bd=None, design_fun=hc_design):
        self.d = d
        self.eta = eta
        self.design_fun = design_fun
        self.input_bd = input_bd
        if input_bd is None:
            self.input_bd = torch.tensor([[0,1]]*d, dtype=torch.float32)
    
    def gen_sg(self):
        d = self.d
        eta = self.eta
        design_fun = self.design_fun
        input_bd = self.input_bd

        # initialize
        x_tot = torch.empty(1,d)
        id_prt = {} # indices of points in this smolyak iteration
        pts_prt = {} # points in this smolyak iteration (each element is d-dimensional llist)
        pts_prt_set = {} # points in this smolyak iteration (each element is [len(t_arrows[prt,:]), d] size tensor)
        
        addpts_prt = {} # points in this smolyak iteration (each element is d-dimensional llist)
        addpts_prt_set = {} # points in this smolyak iteration (each element is [len(t_arrows[prt,:]), d] size tensor)

        ii = 0

        t_sum_start = max(d, eta-d+1)
        for t_sum in range(t_sum_start, eta+1):
            
            t_arrows = n_sum_k(d,t_sum) # [n_prt, d] size tensor
            n_prt = t_arrows.shape[0]
            
            for prt in range(n_prt): # loop over partitions of eta(differnet t_arrow for the same eta)
                x_sg = [0]*d # sparse grid points, d-dimensional list
                xadd_sg = [0]*d # added sparse grid points compared with last level, d-dimensional list
                for dim in range(d): # loop over dimension
                    x_sg[dim] = design_fun( t_arrows[prt,dim], input_bd[dim,:] )
                    xlast_sg = design_fun( t_arrows[prt,dim]-1, input_bd[dim,:] )

                    combined = torch.cat((x_sg[dim], xlast_sg))
                    uniques, counts = combined.unique(return_counts=True)
                    difference = uniques[counts == 1]
                    #intersection = uniques[counts > 1]
                    xadd_sg[dim] = difference
                
                x_prt = torch.cartesian_prod(*x_sg) # [len(t_arrows[prt,:]), d] size tensor
                pts_prt[t_sum, prt] = x_sg # d-dimensional list
                pts_prt_set[t_sum, prt] = x_prt # [len(t_arrows[prt,:]), d] size tensor (full grid poitns at each iteration)
                
                xadd_prt = torch.cartesian_prod(*xadd_sg) # [len(t_arrows[prt,:]), d] size tensor
                addpts_prt[t_sum, prt] = xadd_sg # d-dimensional list
                addpts_prt_set[t_sum, prt] = xadd_prt # [len(t_arrows[prt,:]), d] size tensor (full grid poitns at each iteration)

                x_tot = torch.vstack(( x_tot, x_prt )) # set of all points including same points
                id_prt[t_sum, prt] = torch.arange(ii, ii + x_prt.shape[0]) # [ii : ii + n_prt]
                ii += x_prt.shape[0]
        
        self.pts_tot = x_tot[1:,:]
        self.id_prt = id_prt # use self.pts_tot[ id_x_prt[t_sum, prt], : ] to extract grid points in each smolyak iter
        self.pts_prt = pts_prt # use self.pts_prt[t_sum, prt] to extract a d-dimensional list, each entry is one-dim points forming the sgdesign
        self.pts_prt_set = pts_prt_set # use self.pts_prt_set[t_sum, prt] to extract a [len(t_arrows[prt,:]), d] size tensor
        
        self.addpts_prt = addpts_prt # use self.pts_prt[t_sum, prt] to extract a d-dimensional list, each entry is one-dim points forming the sgdesign
        self.addpts_prt_set = addpts_prt_set # use self.pts_prt_set[t_sum, prt] to extract a [len(t_arrows[prt,:]), d] size tensor

        self.pts_set = torch.unique(self.pts_tot, dim=0)
        self.n_pts = self.pts_set.shape[0]
        return self