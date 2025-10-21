import torch


def simulated_binaryF(x: torch.Tensor, pro_c: float = 1.0, dis_c: float = 20.0) -> torch.Tensor:

    select_pop = x
    # 交叉变异
    N, n_target = select_pop.shape
    p2_indices = torch.randint(0, N, (N,))
    p2_decs = select_pop[p2_indices]  # (N, n_target)

    cross_mask = torch.rand(N, n_target) < 20 / n_target
    offspring = torch.where(cross_mask, p2_decs, select_pop)
    offspring_dec = torch.round(offspring)
    return offspring_dec
