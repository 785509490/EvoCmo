import math
from typing import Callable, Optional

import torch

from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary_half, DE_crossover
from evox.operators.mutation import polynomial_mutation
from evox.operators.sampling import uniform_sampling
from evox.utils import clamp, minimum
from evox.operators.selection import nd_environmental_selection_cons


def pbi(f: torch.Tensor, w: torch.Tensor, z: torch.Tensor):
    norm_w = torch.linalg.norm(w, dim=1)
    f = f - z

    d1 = torch.sum(f * w, dim=1) / norm_w

    d2 = torch.linalg.norm(f - (d1[:, None] * w / norm_w[:, None]), dim=1)
    return d1 + 5 * d2

def tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) * w, dim=1)[0]


def tchebycheff_norm(f, w, z, z_max):
    return torch.max(torch.abs(f - z) / (z_max - z) * w, dim=1)[0]


def modified_tchebycheff(f, w, z):
    return torch.max(torch.abs(f - z) / w, dim=1)[0]
class PPS2(Algorithm):

    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        aggregate_op=("tchebycheff", "tchebycheff"),
        max_gen: int = 100,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        """Initializes the MOEA/D algorithm.

        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions in the optimization problem.
        :param lb: The lower bounds for the decision variables (1D tensor).
        :param ub: The upper bounds for the decision variables (1D tensor).
        :param selection_op: The selection operation for evolutionary strategy (optional).
        :param mutation_op: The mutation operation, defaults to `polynomial_mutation` if not provided (optional).
        :param crossover_op: The crossover operation, defaults to `simulated_binary_half` if not provided (optional).
        :param device: The device on which computations should run (optional). Defaults to PyTorch's default device.
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        if device is None:
            device = torch.get_default_device()
        # check
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]
        # write to self
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op
        self.device = device

        if self.mutation is None:
            self.mutation = polynomial_mutation
        if self.crossover is None:
            self.crossover = simulated_binary_half

        w, _ = uniform_sampling(self.pop_size, self.n_objs)

        self.pop_size = w.size(0)
        self.n_neighbor = int(math.ceil(self.pop_size / 10))

        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        neighbors = torch.cdist(w, w)
        self.neighbors = torch.argsort(neighbors, dim=1, stable=True)[:, : self.n_neighbor]
        self.w = w

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.cons = None
        self.ideal_points = Mutable(torch.zeros((1, self.n_objs), device=device))
        self.nadir_points = Mutable(torch.zeros((1, self.n_objs), device=device))
        self.z = Mutable(torch.zeros((self.n_objs,), device=device))
        self.gen = 0
        self.last_gen = 20
        self.max_gen = max_gen
        self.Tc  = 0.9 * self.max_gen
        self.change_threshold = 1e-1
        self.search_stage = 1
        self.max_change = 1
        self.epsilon_k = 0
        self.epsilon_0 = 0
        self.cp = 2
        self.alpha = 0.95
        self.tao = 0.05

        self.archpop = self.pop
        self.archfit = self.fit
        self.archcons = self.cons
    def get_aggregation_function(self, name: str) -> Callable:
        aggregation_functions = {
            "pbi": pbi,
            "tchebycheff": tchebycheff,
            "tchebycheff_norm": tchebycheff_norm,
            "modified_tchebycheff": modified_tchebycheff,
        }
        if name not in aggregation_functions:
            raise ValueError(f"Unsupported function: {name}")
        return aggregation_functions[name]

    def init_step(self):
        fitness = self.evaluate(self.pop)
        if isinstance(fitness, tuple):
            self.fit = fitness[0]
            self.cons = fitness[1]
            self.archpop = self.pop
            self.archfit = self.fit
            self.archcons = self.cons
        else:
            self.fit = fitness
        self.z = torch.min(self.fit, dim=0)[0]
    @staticmethod
    def calc_maxchange(ideal_points, nadir_points, gen, last_gen):
        delta_value = 1e-6 * torch.ones(1, ideal_points.size(1), device=ideal_points.device)
        rz = torch.abs((ideal_points[gen, :] - ideal_points[gen - last_gen + 1, :]) / torch.max(ideal_points[gen - last_gen + 1, :], delta_value))
        nrz = torch.abs((nadir_points[gen, :] - nadir_points[gen - last_gen + 1, :]) / torch.max(nadir_points[gen - last_gen + 1, :], delta_value))
        return torch.max(torch.cat((rz, nrz), dim=0))

    @staticmethod
    def update_epsilon(tao, epsilon_k, epsilon_0, rf, alpha, gen, Tc, cp):
        if rf < alpha:
            return (1 - tao) * epsilon_k
        else:
            return epsilon_0 * ((1 - (gen / Tc)) ** cp)

    def step(self):
        """Perform a single optimization step of the workflow."""
        for i in range(self.pop_size):
            parents = self.neighbors[i][torch.randperm(self.n_neighbor, device=self.device)]
            if self.crossover is DE_crossover:
                CR = torch.ones((self.pop_size, self.dim))
                F = torch.ones((self.pop_size, self.dim)) * 0.5
                crossovered = self.crossover(self.pop[parents[0]], self.pop[parents[1]],
                                             self.pop[parents[2]], CR, F)
            else:
                crossovered = self.crossover(self.pop[parents[:2]])
            offspring = self.mutation(crossovered, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)
            off_fit = self.evaluate(offspring)

            off_cons = off_fit[1]
            off_fit = off_fit[0]
            cv = torch.sum(torch.clamp(self.cons, min=0), dim=1, keepdim=True)
            cv_off = torch.sum(torch.clamp(off_cons, min=0), dim=1, keepdim=True)
            rf = (cv <= 1e-6).sum().item() / self.pop_size

            temp = torch.cat([self.pop, self.fit, cv], dim=1)
            self.z = torch.min(self.z, torch.min(off_fit, dim=0)[0])
            self.z = minimum(self.z, off_fit)
            if self.gen == 0:
                self.ideal_points[0, :] = self.z
            else:
                self.ideal_points = torch.cat([self.ideal_points, self.z], dim=0)

            D = self.pop.size(1)
            M = self.fit.size(1)
            if self.gen == 0:
                self.nadir_points[0, :] = torch.max(temp[:, D:D + M], dim=0)[0]
            else:
                b = torch.max(temp[:, D:D + M], dim=0)[0].unsqueeze(0)
                self.nadir_points = torch.cat([self.nadir_points, b], dim=0)

            if self.gen >= self.last_gen:
                self.max_change = self.calc_maxchange(self.ideal_points, self.nadir_points, self.gen, self.last_gen)
            # The value of e(k) and the search strategy are set.
            if self.gen < self.Tc:
                if self.max_change <= self.change_threshold and self.search_stage == 1:
                    self.search_stage = -1
                    self.epsilon_0 = temp[:, -1].max().item()
                    self.epsilon_k = self.epsilon_0
                if self.search_stage == -1:
                    self.epsilon_k = self.update_epsilon(self.tao, self.epsilon_k, self.epsilon_0, rf, self.alpha,
                                                         self.gen, self.Tc, self.cp)
            else:
                self.epsilon_k = 0


            g_old = pbi(self.fit[parents], self.w[parents], self.z)
            g_new = pbi(off_fit, self.w[parents], self.z)
            cv_old = torch.sum(torch.clamp(self.cons[parents], min=0), dim=1, keepdim=True)
            cv_new = torch.sum(torch.clamp(off_cons, min=0), dim=1, keepdim=True)
            if self.search_stage == 1:
                self.fit[parents[g_old >= g_new]] = off_fit
                self.pop[parents[g_old >= g_new]] = offspring
                self.cons[parents[g_old >= g_new]] = off_cons
            else:
                condition = ((g_old > g_new) & (((cv_old <= self.epsilon_k) & (cv_new <= self.epsilon_k)) | (cv_old == cv_new)) | (cv_new < cv_old) )
                self.fit[parents[g_old >= g_new]] = off_fit
                self.pop[parents[g_old >= g_new]] = offspring
                self.cons[parents[g_old >= g_new]] = off_cons

        self.gen += 1
        merge_pop = torch.cat([self.archpop, self.pop], dim=0)
        merge_fit = torch.cat([self.archfit, self.fit], dim=0)
        merge_cons = torch.cat([self.archcons, self.cons], dim=0)
        self.archpop, self.archfit, _, _, self.archcons = nd_environmental_selection_cons(merge_pop, merge_fit, merge_cons, self.pop_size)
        if self.gen >= self.max_gen:
            self.pop, self.fit, _, _, self.cons = nd_environmental_selection_cons(merge_pop, merge_fit, merge_cons, self.pop_size)
