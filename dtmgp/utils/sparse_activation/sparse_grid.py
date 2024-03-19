import torch
import math
from dtmgp.utils.sparse_activation.design_class import HyperbolicCrossDesign


class SparseGridDesign:
    """
    :class:`~SparseGridDeisgn` is a class Generate a sparse grid structure of dimension :math:`d`, level :math:`\eta`
    and base design class

    :param d: dimension of input, d >= 2
    :type d: int
    :param eta: (Default: d+2) level of sparse grid design, eta > d
    :type eta: int
    :param, input_bd: (Default: [[0,1]]*d)
    :type input_bd: [d,2] size list
    :param design_class: design used in one dimension, default = hc_design (Here we assume the sparse grid design is the same in each dimension)
    :type design_class: class
    ...
    :raise RuntimeError: level eta should be greater than dimension d
    

    Example:
        >>> sgdesign_init = SparseGridDesign(d=2, eta=4, input_bd=input_bd, design_class=HyperbolicCrossDesign)
        >>> sgdesign = sgdesign_init.gen_sg(dyadic_sort=True, return_neighbors=True)
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
        """
        ...
        :return: an object with sparse grid design
        :rtype: class
        """
        d = self.d
        eta = self.eta
        design_class = self.design_class
        input_bd = self.input_bd

        # initialize
        x_tot = torch.empty(1,d)
        indices_tot = torch.empty(1,d,dtype=int)

        level_combs = {}

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
            level_combs[t_sum] = t_arrows
            
            for prt in range(n_prt): # loop over partitions of eta(differnet t_arrow for the same eta)
                design_str_fg = [0]*d
                x_fg = [0]*d # sparse grid points, d-dimensional list
                indices_fg = [0]*d # sparse grid points, d-dimensional list
                
                for dim in range(d): # loop over dimension
                    design_fun = design_class(dyadic_sort=dyadic_sort, return_neighbors=return_neighbors)
                    design_str = design_fun( t_arrows[prt,dim], input_bd[dim])
                    design_str_fg[dim] = design_str
                    x_fg[dim] = design_str.points
                    indices_fg[dim] = (2**(eta-d+1) * design_str.points - 1).to(dtype=int)
                
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
        
        self.level_combs = level_combs

        # obtain the set of pts_tot and preserve the order (remove duplicates in pts_tot)
        pts_tot_items = [tuple(l) for l in self.pts_tot.tolist()]
        pts_set_list = list(dict.fromkeys(pts_tot_items))
        self.pts_set = torch.tensor(pts_set_list, dtype=torch.float32)
        self.n_pts = self.pts_set.shape[0]

        self.design_str_prt = design_str_prt

        self.indices_prt = indices_prt
        self.indices_prt_set = indices_prt_set
        self.indices_tot = indices_tot[1:,:]
        indices_tot_items = [tuple(l) for l in self.indices_tot.tolist()]
        indices_set_list = list(dict.fromkeys(indices_tot_items))
        self.indices_set = torch.tensor(indices_set_list, dtype=int)
        
        return self
    

def n_sum_k(n, k):
    """
    Method ref:
    https://www.mathworks.com/matlabcentral/fileexchange/28340-nsumk
    https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)

    Parameters:
    ------------------------
    n: # of positive integer
    k: sum of the integers = k

    Returns:
    ------------------------
    a tensor of all possible combinations of n positive integers adding up to a given number k 
    """
    if n == 1:
        return torch.tensor( [[k]] )
    else:
        num_comb = math.comb(k-1, n-1)
        d1 = torch.zeros(num_comb, 1, dtype=torch.int64) # [num_comb] size tensor
        sum_vec = torch.arange(1, k)
        d2 = torch.combinations(sum_vec, r=n-1) # [num_comb, n-1] size tensor
        d3 = torch.ones(num_comb, 1, dtype=torch.int64) * k # [num_comb] size tensor

        dividers = torch.cat((d1, d2, d3), dim=1)
        res = torch.diff(dividers, dim=1) # [num_comb, n] size tensor
        return res