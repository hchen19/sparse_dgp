import torch
from dtmgp.utils.operators.chol_inv import n_sum_k
from dtmgp.utils.sparse_activation.design_class import HyperbolicCrossDesign


class SparseGridDesign:
    """
    ------------------------
    Parameters:
    d: dimension of input, d >= 2
    eta: level of sparse grid design
    design_fun: design used in one dimension, default = hc_design (Here we assume the sparse grid design is the same in each dimension)
    input_bd: [d,2] size list, default value = [[0,1]]*d

    ------------------------
    Returns:
    self: an object with sparse grid design
    """

    def __init__(self, d, eta=None, 
                 input_bd=None, 
                 design_class=HyperbolicCrossDesign
                 ):
        self.d = d
        self.eta = eta
        self.design_class = design_class
        self.input_bd = input_bd
        if eta is None:
            eta = d + 2
        if input_bd is None:
            self.input_bd = [[0,1]]*d
            #self.input_bd = torch.tensor([[0,1]]*d, dtype=torch.float32)
        if d >= eta:
            raise RuntimeError("level eta should be greater than dimension d")
    
    def gen_sg(self,dyadic_sort=True, return_neighbors=True):
        d = self.d
        eta = self.eta
        design_class = self.design_class
        input_bd = self.input_bd

        # initialize
        x_tot = torch.empty(1,d)
        indices_tot = torch.empty(1,d,dtype=int)
        id_prt = {} # indices of points in this smolyak iteration
        pts_prt = {} # points in this smolyak iteration (each element is d-dimensional llist)
        pts_prt_set = {} # points in this smolyak iteration (each element is [len(t_arrows[prt,:]), d] size tensor)
        
        design_str_prt = {}
        indices_prt = {}
        indices_prt_set = {}

        ii = 0
        t_sum_start = max(d, eta-d+1)
        
        for t_sum in range(t_sum_start, eta+1):
            
            t_arrows = n_sum_k(d,t_sum) # [n_prt, d] size tensor
            n_prt = t_arrows.shape[0]
            
            for prt in range(n_prt): # loop over partitions of eta(differnet t_arrow for the same eta)
                design_str_fg = [0]*d
                x_fg = [0]*d # sparse grid points, d-dimensional list
                indices_fg = [0]*d # sparse grid points, d-dimensional list
                
                for dim in range(d): # loop over dimension
                    design_fun = design_class(dyadic_sort=dyadic_sort, return_neighbors=return_neighbors)
                    design_str = design_fun( t_arrows[prt,dim], input_bd[dim])
                    design_str_fg[dim] = design_str
                    x_fg[dim] = design_str.points
                    indices_fg[dim] = (2**(eta-1) * design_str.points - 1).to(dtype=int)
                
                # design structure
                design_str_prt[t_sum, prt] = design_str_fg

                # indices of points
                indices_prt_sg = torch.cartesian_prod(*indices_fg)
                indices_prt[t_sum, prt] = indices_fg
                indices_prt_set[t_sum, prt] = indices_prt_sg

                # points
                x_prt = torch.cartesian_prod(*x_fg) # [len(t_arrows[prt,:]), d] size tensor
                pts_prt[t_sum, prt] = x_fg # d-dimensional list
                pts_prt_set[t_sum, prt] = x_prt # [len(t_arrows[prt,:]), d] size tensor (full grid poitns at each iteration)
                
                if t_sum == eta:
                    indices_tot = torch.vstack(( indices_tot, indices_prt_sg ))
                    x_tot = torch.vstack(( x_tot, x_prt )) # set of all points including same points
                    id_prt[t_sum, prt] = torch.arange(ii, ii + x_prt.shape[0]) # [ii : ii + n_prt]
                    ii += x_prt.shape[0]
        
        self.pts_tot = x_tot[1:,:]
        self.id_prt = id_prt # use self.pts_tot[ id_x_prt[t_sum, prt], : ] to extract grid points in each smolyak iter
        self.pts_prt = pts_prt # use self.pts_prt[t_sum, prt] to extract a d-dimensional list, each entry is one-dim points forming the sgdesign
        self.pts_prt_set = pts_prt_set # use self.pts_prt_set[t_sum, prt] to extract a [len(t_arrows[prt,:]), d] size tensor
        
        # obtain the set of pts_tot and preserve the order (remove duplicates in pts_tot)
        pts_tot_items = [tuple(l) for l in self.pts_tot.tolist()]
        pts_set_list = list(dict.fromkeys(pts_tot_items))
        self.pts_set = torch.tensor(pts_set_list, dtype=float)
        self.n_pts = self.pts_set.shape[0]

        self.design_str_prt = design_str_prt

        self.indices_prt = indices_prt
        self.indices_prt_set = indices_prt_set
        self.indices_tot = indices_tot[1:,:]
        indices_tot_items = [tuple(l) for l in self.indices_tot.tolist()]
        indices_set_list = list(dict.fromkeys(indices_tot_items))
        self.indices_set = torch.tensor(indices_set_list, dtype=int)
        
        return self