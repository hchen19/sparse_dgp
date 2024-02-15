from typing import Optional
import torch

class HyperbolicCrossDesign:
    def __init__(self, 
                 dyadic_sort: Optional[bool] = True, 
                 indices_sort: Optional[bool] = False) -> None:
        """
        ------------------------
        Parameters:
        dyadic_sort: if sort=True, return sorted incremental tensor, default=True
        indices_sort: whether to return the indicies_sort of dyadic_points
                Default = False
        ------------------------
        Returns:
        None
        """
        self.dyadic_sort = dyadic_sort
        self.indices_sort = indices_sort
    
    def __call__(self, deg, input_bd = [0,1]):
        """
        ------------------------
        Parameters:
        deg: degree of hyperbolic cross (# of points = 2^deg - 1)
        input_bd: [x_leftbd, x_rightbd], [2] size list
        dyadic_sort: if sort=True, return sorted incremental tensor, default=True
        

        ------------------------
        Returns:
        pts: [1/(2^deg), 2/(2^deg), 3/(2^deg),..., (2^deg-1)/(2^deg)]
            [2^deg-1] size tensor with hyperbolic cross points (bisection)        
        """
        x_1 = input_bd[0]
        x_n = input_bd[1]

        if self.dyadic_sort is True:
            res_basis = torch.empty(0)
            for i in range(1, deg+1):
                increment_set = torch.arange(start=1, end=2**i, step=2) * (0.5**i)
                res_basis = torch.cat((res_basis,increment_set),dim=0)
        else:
            res_basis = torch.arange(start=1, end=2**deg, step=1) * (0.5**deg) # interval on [0,1]

        res = res_basis**(x_n-x_1) + x_1
        self.points = res
        
        if self.indices_sort is True:
            # indices of the sorted_dyadic
            sorted_dyadic, indices_dyadic = torch.sort(res)
            indices_sort = torch.argsort(indices_dyadic)
            self.sorted_vals_indices = sorted_dyadic, indices_dyadic
            self.indices_sort = indices_sort
        
        return self