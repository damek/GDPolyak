import torch
import torch.autograd.functional
from torch.optim.optimizer import Optimizer
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsmr, lsqr
from typing import Callable, Iterable, Optional
import linops
from enum import Enum



class GDPolyak(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            GD_step_size,
            nb_GD_steps, 
            nb_polyak_steps, 
            nb_restarts,
            opt_est,
            grad_size_tol=1e-5,
            print_freq=10,
            distance_func: Optional[Callable] = None,
            polyak_increase_factor=1.001
    ):
        super(GDPolyak, self).__init__(params, dict())

        self._params = self.param_groups[0]["params"]
        self._numel = sum([p.numel() for p in self._params])
        self.tol = grad_size_tol
        self.GD_step_size = GD_step_size
        self.opt_est = opt_est
        self.nb_GD_steps = int(nb_GD_steps)
        self.nb_polyak_steps = nb_polyak_steps
        self.nb_restarts = nb_restarts
        self.min_loss_value = np.inf
        self.iteration_counter = 0
        self.history_loss = []
        self.history_dist_to_opt_solution = []
        self.step_size_list = []
        self.print_freq = print_freq
        self.distance_func = distance_func
        self.oracle_calls = None

  
    
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel

    def _gather_flat_grad(self) -> np.ndarray:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _gather_flat_params(self) -> np.ndarray:
        views = []
        for p in self._params:
            view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @torch.no_grad()
    def step(self, closure: Callable, step_size):
        
        self.step_size_list.append(step_size)
        # allow closure to call autograd
        with torch.enable_grad():
            # evaluate closure
            loss = closure().item()
            
        g = self._gather_flat_grad()

        # Update min loss value seen so far
        if loss < self.min_loss_value:
            self.min_loss_value = loss

        self._add_grad(-step_size, g)
        # Update the loss history, handling NaN values
        if not np.isnan(loss):
            self.history_loss.append(loss)
        else:
            # If loss is NaN, use the previous value or a default value
            if self.history_loss:
                self.history_loss.append(self.history_loss[-1])
            else:
                self.history_loss.append(float('inf'))  # Use infinity as a default value
        if self.distance_func is not None:
            distance = self.distance_func()
            if np.isnan(distance):
                if self.history_dist_to_opt_solution:
                    distance = self.history_dist_to_opt_solution[-1]
                else:
                    distance = float('inf')  # Use infinity as a default value if no previous distance
            self.history_dist_to_opt_solution.append(distance)
        self.iteration_counter += 1
        if self.iteration_counter % self.print_freq == 0:
            print(f"Iteration {self.iteration_counter}: loss = {loss}")
        if self.distance_func is not None and self.iteration_counter % self.print_freq == 0:
            print(f"Iteration {self.iteration_counter}: distance to opt sol = {self.history_dist_to_opt_solution[-1]}")

    @torch.no_grad()
    def run_GDLoop(self, closure: Callable):
        for i in range(self.nb_GD_steps):
            self.step(closure, self.GD_step_size)
            with torch.enable_grad():
                cur_loss = closure().item()
            if np.isnan(cur_loss):
                print("NaN loss value encountered. Stopping GDPolyak.")
                break


    @torch.no_grad()
    def run_GDPolyak(self, closure: Callable):
        initial_params = [p.clone().detach() for p in self._params] if self.nb_restarts > 1 else None
        for _ in range(self.nb_restarts):
            if self.nb_restarts > 1:
                for p, initial_p in zip(self._params, initial_params):
                    p.copy_(initial_p)
            if self.nb_polyak_steps > 0:
                for _ in range(self.nb_polyak_steps):
                    self.run_GDLoop(closure) 
                    # current loss value
                    with torch.enable_grad():
                        cur_loss = closure().item()
                    if np.isnan(cur_loss):
                        print("NaN loss value encountered. Stopping GDPolyak.")
                        break
                    g = self._gather_flat_grad()
                    if g.norm() == 0:
                        print("Gradient is zero. Stopping GDPolyak.")
                        break
                    polyak_step_size = (cur_loss - self.opt_est) / (g.norm() ** 2).item()
                    if self.nb_restarts > 1:
                        polyak_step_size /= 2
                    self.step(closure, polyak_step_size)
                self.opt_est = (self.min_loss_value + self.opt_est)/2
            else: 
                self.run_GDLoop(closure) 

                

   

def run_gdpolyak_algorithm(loss_fn, 
                           params, 
                           nb_restarts, 
                           nb_polyak_steps, 
                           nb_gd_steps, 
                           gd_step_size,
                           opt_est, 
                           distance_func=None, 
                           print_freq=100):
    # Define the GDPolyak optimizer
    optimizer = GDPolyak(
        params,
        nb_restarts=nb_restarts,
        nb_polyak_steps=nb_polyak_steps,
        nb_GD_steps=nb_gd_steps,
        GD_step_size=gd_step_size,
        opt_est=opt_est,
        distance_func=distance_func,
        print_freq=print_freq
    )

    # Define the closure
    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    # Run the GDPolyak algorithm
    optimizer.run_GDPolyak(closure)

    # Return the history
    return {
        'loss': optimizer.history_loss,
        'step_sizes': optimizer.step_size_list,
        'dist_to_opt': optimizer.history_dist_to_opt_solution if distance_func is not None else []
    }

