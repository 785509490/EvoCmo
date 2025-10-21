import torch

from evox.utils import maximum, minimum


def polynomial_mutationF(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    pro_m: float = 1,
    dis_m: float = 20,
) -> torch.Tensor:
    select_pop = x
    # 交叉变异
    N, n_target = select_pop.shape
    offspring = select_pop
    mutate_mask = torch.rand(N, n_target) < 10 / n_target
    rule_counts = ub
    random_picks = (torch.rand((N, n_target)) * rule_counts).long()
    random_picks = random_picks.to(dtype=offspring.dtype)
    offspring = torch.where(mutate_mask, random_picks, offspring)
    offspring_dec = torch.round(offspring)
    return offspring_dec
