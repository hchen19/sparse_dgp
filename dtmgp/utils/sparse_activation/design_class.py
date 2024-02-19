from typing import Optional
import torch

class HyperbolicCrossDesign:
    def __init__(self, 
                 dyadic_sort: Optional[bool] = True, 
                 return_neighbors: Optional[bool] = False) -> None:
        """
        ------------------------
        Parameters:
        dyadic_sort: if sort=True, return sorted incremental tensor, default=True
        return_neighbors: whether to return the neighbors of dyadic_points
                Default = False
        
        ------------------------
        Returns:
        None
        """
        self.dyadic_sort = dyadic_sort
        self.return_neighbors = return_neighbors
    
    def __call__(self, deg, input_bd = [0,1]):
        """
        ------------------------
        Parameters:
        deg: degree of hyperbolic cross (# of points = 2^deg - 1)
        input_bd: [x_leftbd, x_rightbd], [2] size list

        ------------------------
        Returns:
        pts: [1/(2^deg), 2/(2^deg), 3/(2^deg),..., (2^deg-1)/(2^deg)]
            [2^deg-1] size tensor with hyperbolic cross points (bisection)        
        """
        x_1 = input_bd[0]
        x_n = input_bd[1]

        if self.dyadic_sort is False and self.return_neighbors is True:
            self.return_neighbors = False
            raise Warning("indices_sort is set to False because dyadic_sort = False")

        if self.dyadic_sort is True:
            
            res_basis = torch.empty(0)
            xlefts = torch.empty(0)
            xrights = torch.empty(0)
            ii = 0
            
            if deg == 0:
                res = torch.empty(0)
            else:    
                for i in range(1, deg+1):
                    increment_set = torch.arange(start=1, end=2**i, step=2) * (0.5**i)
                    res_basis = torch.cat((res_basis,increment_set),dim=0)
                    res = res_basis*(x_n-x_1) + x_1
                    len_increment = len(increment_set)
                    if self.return_neighbors is True:
                        
                        # indices of the sorted_dyadic
                        sorted_dyadic, indices_dyadic = torch.sort(res)
                        indices_sort = torch.argsort(indices_dyadic)
                        if i == deg:
                            self.indices_sort = indices_sort
                        
                        # indices of closest left neighbors and right neighbors
                        left_indices = indices_sort - 1
                        right_indices = indices_sort + 1

                        # search the closest left neighbor xleft and right neighbor xright
                        sorted_dyadic_extend = torch.cat(( torch.tensor([float('-inf')]), 
                                                        sorted_dyadic, 
                                                        torch.tensor([float('inf')]) ), 
                                                        dim=0)
                    

                        increment_xlefts = sorted_dyadic_extend[left_indices+1][ii : ii+len_increment]
                        increment_xrights = sorted_dyadic_extend[right_indices+1][ii : ii+len_increment]
                        
                        xlefts = torch.cat((xlefts,increment_xlefts),dim=0)
                        xrights = torch.cat((xrights,increment_xrights),dim=0)
                    
                    ii += len_increment
        
        else:
            res_basis = torch.arange(start=1, end=2**deg, step=1) * (0.5**deg) # interval on [0,1]
            res = res_basis*(x_n-x_1) + x_1

        self.points = res
        if self.return_neighbors is True:
            self.lefts = xlefts
            self.rights = xrights
        
        return self