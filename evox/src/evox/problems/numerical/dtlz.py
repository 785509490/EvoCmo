import torch

from evox.core import Problem
from evox.operators.sampling import grid_sampling, uniform_sampling
import numpy as np

class DTLZ(Problem):
    """
    Base class for DTLZ test suite problems in multi-objective optimization.

    Inherit this class to implement specific DTLZ problem variants.

    :param d: Number of decision variables.
    :param m: Number of objectives.
    :param ref_num: Number of reference points used in the problem.
    """

    def __init__(self, d: int = None, m: int = None, ref_num: int = 1000):
        """Override the setup method to initialize the parameters"""
        super().__init__()
        self.d = d
        self.m = m
        self.ref_num = ref_num
        self.sample, _ = uniform_sampling(self.ref_num * self.m, self.m)  # Assuming UniformSampling is defined
        self.device = self.sample.device

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to evaluate the objective values for given decision variables.

        :param X: A tensor of shape (n, d), where n is the number of solutions and d is the number of decision variables.
        :return: A tensor of shape (n, m) representing the objective values for each solution.
        """
        raise NotImplementedError()

    def pf(self):
        """
        Return the Pareto front for the problem.

        :return: A tensor representing the Pareto front.
        """
        f = self.sample / 2
        return f


class DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        return f


class DTLZ2(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        f = self.sample
        f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f


class DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f


class DTLZ4(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        Xfront = X[:, : m - 1].pow(100)
        Xrear = X[:, m - 1:].clone()
        # X[:, : m - 1] = X[:, : m - 1].pow(100)

        g = torch.sum((Xrear - 0.5) ** 2, dim=1, keepdim=True)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f


class DTLZ5(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        temp = g.repeat(1, m - 2)

        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * Xfront[:, 1:]) / (2 + 2 * temp)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        return f

    def pf(self):
        n = self.ref_num * self.m

        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ6(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] ** 0.1), dim=1, keepdim=True)
        temp = torch.tile(g, (1, m - 2))
        Xfront = X[:, : m - 1].clone()
        Xfront[:, 1:] = (1 + 2 * temp * Xfront[:, 1:]) / (2 + 2 * temp)

        f = (
            torch.tile(1 + g, (1, m))
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        return f

    def pf(self):
        n = self.ref_num * self.m

        # Ensure the tensor is created on the same device (use X.device if needed)
        f = torch.vstack(
            (
                torch.hstack(
                    (
                        torch.arange(0, 1, 1.0 / (n - 1), device=self.device),
                        torch.tensor(1.0, device=self.device),
                    )
                ),
                torch.hstack(
                    (
                        torch.arange(1, 0, -1.0 / (n - 1), device=self.device),
                        torch.tensor(0.0, device=self.device),
                    )
                ),
            )
        ).T

        f = f / torch.tile(torch.sqrt(torch.sum(f**2, dim=1, keepdim=True)), (1, f.size(1)))

        for i in range(self.m - 2):
            f = torch.cat((f[:, 0:1], f), dim=1)

        f = f / torch.sqrt(torch.tensor(2.0, device=self.device)) ** torch.tile(
            torch.hstack(
                (
                    torch.tensor(self.m - 2, device=self.device),
                    torch.arange(self.m - 2, -1, -1, device=self.device),
                )
            ),
            (f.size(0), 1),
        )
        return f


class DTLZ7(DTLZ):
    def __init__(self, d: int = 21, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)
        self.sample, _ = grid_sampling(self.ref_num * self.m, self.m - 1)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        f = torch.zeros((n, m), device=X.device)
        g = 1 + 9 * torch.mean(X[:, m - 1 :], dim=1, keepdim=True)

        term = torch.sum(
            X[:, : m - 1] / (1 + torch.tile(g, (1, m - 1))) * (1 + torch.sin(3 * torch.pi * X[:, : m - 1])),
            dim=1,
            keepdim=True,
        )
        f = torch.cat([X[:, : m - 1].clone(), (1 + g) * (m - term)], dim=1)

        return f

    def pf(self):
        interval = torch.tensor([0.0, 0.251412, 0.631627, 0.859401], dtype=torch.float, device=self.device)
        median = (interval[1] - interval[0]) / (interval[3] - interval[2] + interval[1] - interval[0]).to(self.device)

        x = self.sample.to(self.device)

        mask_less_equal_median = x <= median
        mask_greater_median = x > median

        x = torch.where(
            mask_less_equal_median,
            x * (interval[1] - interval[0]) / median + interval[0],
            x,
        )
        x = torch.where(
            mask_greater_median,
            (x - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
            x,
        )

        last_col = 2 * (self.m - torch.sum(x / 2 * (1 + torch.sin(3 * torch.pi * x)), dim=1, keepdim=True))

        pf = torch.cat([x, last_col], dim=1)
        return pf


class C1_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> tuple:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        PopCon = (f[:, -1].unsqueeze(1) / 0.6) + (torch.sum(f[:, :-1] / 0.5, dim=1, keepdim=True)) - 1
        return f, PopCon


class C2_DTLZ2(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        g = torch.sum((X[:, m - 1 :] - 0.5) ** 2, dim=1, keepdim=True)
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((X.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((X.size(0), 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        if m == 3:
            r = 0.4
        else:
            r = 0.5
        constraint1 = torch.min(
            (PopObj - 1) ** 2 + torch.sum(PopObj ** 2, dim=1, keepdim=True) - PopObj ** 2 - r ** 2,
            dim=1,
            keepdim=True
        )[0]
        constraint2 = torch.sum((PopObj - 1 / torch.sqrt(torch.tensor(m))) ** 2, dim=1, keepdim=True) - r ** 2
        PopCon = torch.min(constraint1, constraint2)
        return PopObj, PopCon

    def GetOptimum(self):
        R = self.sample
        R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True))

        if self.m == 3:
            r = 0.4
        else:
            r = 0.5

        mask = (np.min((R - 1) ** 2 + np.sum(R ** 2, axis=1, keepdims=True) - R ** 2 - r ** 2, axis=1) > 0) & \
               (np.sum((R - 1 / np.sqrt(self.M)) ** 2, axis=1) - r ** 2 > 0)

        R = R[~mask]

        return R

    def pf(self):
        if self.m == 2:
            f = self.sample
            f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
            return f

        elif self.m == 3:
            a = np.linspace(0, np.pi / 2, 60)
            x = np.outer(np.sin(a), np.cos(a))
            y = np.outer(np.sin(a), np.sin(a))
            z = np.outer(np.cos(a), np.ones_like(a))

            R = np.column_stack((x.flatten(), y.flatten(), z.flatten()))

            fes = (np.min((R - 1) ** 2 + np.sum(R ** 2, axis=1, keepdims=True) - R ** 2 - 0.4 ** 2, axis=1) <= 0) | \
                  (np.sum((R - 1 / np.sqrt(3)) ** 2, axis=1) - 0.4 ** 2 <= 0)
            valid_indices = np.where(fes)[0]
            x_filtered = x.flatten()[valid_indices]
            y_filtered = y.flatten()[valid_indices]
            z_filtered = z.flatten()[valid_indices]

            result = np.column_stack((x_filtered, y_filtered, z_filtered))
            return torch.tensor(result, dtype=torch.float32)

        else:
            return torch.tensor([])

class C1_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        if self.m == 2:
            r = 6
        elif self.m <= 3:
            r = 9
        elif self.m <= 8:
            r = 12.5
        else:
            r = 15
        PopCon = -(torch.sum(PopObj ** 2, dim=1, keepdim=True) - 16) * (torch.sum(PopObj ** 2, dim=1, keepdim=True) - r ** 2)

        return PopObj, PopCon

class C3_DTLZ4(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m

        Xfront = X[:, : m - 1].pow(100)
        Xrear = X[:, m - 1:].clone()

        g = torch.sum((Xrear - 0.5) ** 2, dim=1, keepdim=True)

        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((g.size(0), 1), device=X.device),
                            torch.maximum(
                                torch.cos(Xfront * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((g.size(0), 1), device=X.device),
                    torch.sin(torch.flip(Xfront, dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        PopObj = f
        PopCon = 1 - PopObj ** 2 / 4 - (torch.sum(PopObj ** 2, dim=1, keepdim=True) - PopObj ** 2)

        return PopObj, PopCon

    def GetOptimum(self):
        f = self.sample
        f = f / torch.sqrt(f.pow(2).sum(dim=1, keepdim=True))
        return f

    def pf(self):
        if self.m == 2:
            R = self.GetOptimum(100)
            return R

        elif self.m == 3:
            a = np.linspace(0, np.pi / 2, 60)
            x = np.outer(np.sin(a), np.cos(a))
            y = np.outer(np.sin(a), np.sin(a))
            z = np.outer(np.cos(a), np.ones_like(a))
            R = np.column_stack((x.flatten(), y.flatten(), z.flatten()))  # Shape (100, 3)
            R = R / np.sqrt(np.sum(R ** 2, axis=1, keepdims=True) - 3 / 4 * np.max(R ** 2, axis=1, keepdims=True))
            result = [R[:, 0].reshape(x.shape), R[:, 1].reshape(x.shape), R[:, 2].reshape(x.shape)]
            return torch.tensor(R, dtype=torch.float32)

        else:
            return torch.tensor([])

class DC1_DTLZ1(DTLZ):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part

        pop_con = 0.5 - torch.cos(3 * torch.pi * X[:, 0])
        return f, pop_con.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            R = torch.zeros((100, 2), dtype=torch.float32)
            R[:, 0] = torch.linspace(0, 1, 100)
            R[:, 1] = 1 - R[:, 0]
            R[torch.cos(3 * torch.pi * R[:, 0]) < 0.5, :] = float('nan')
            R /= 2
            R = R[~torch.isnan(R).any(dim=1)]
            return R

        elif self.m == 3:
            a = torch.linspace(0, 1, 60).view(-1, 1)
            x = a @ a.T
            y = a * (1 - a.T)
            z = (1 - a) * torch.ones_like(x)
            mask = torch.cos(3 * torch.pi * a) < 0.5
            mask = mask.expand_as(z)
            z[mask] = float('nan')

            # 合并并返回结果
            R = torch.cat([(x / 2).view(-1, 1), (y / 2).view(-1, 1), (z / 2).view(-1, 1)], dim=1)
            R = R[~torch.isnan(R).any(dim=1)]
            return R

        else:
            return torch.tensor([])

class DC1_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        pop_con = 0.5 - torch.cos(3 * torch.pi * X[:, 0])
        return f, pop_con.unsqueeze(1)

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            x = torch.linspace(0, torch.pi / 2, 100).view(-1, 1)
            R = torch.zeros((100, 2), dtype=torch.float32)
            R[:, 0] = torch.cos(x).squeeze()
            R[:, 1] = torch.sin(x).squeeze()
            R[torch.cos(6 * x) < 0.5, :] = float('nan')
            R = R[~torch.isnan(R).any(dim=1)]
            return R

        elif self.m == 3:
            a = torch.linspace(0, torch.pi / 2, 50).view(-1, 1)
            x = torch.cos(a) @ torch.cos(a.T)
            y = torch.cos(a) @ torch.sin(a.T)
            z = torch.sin(a) * torch.ones_like(x)

            mask = torch.cos(6 * a) < 0.5
            z[mask.expand_as(z)] = float('nan')
            R = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), dim=1)
            R = R[~torch.isnan(R).any(dim=1)]
            return R

        else:
            return torch.tensor([])


class DC2_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1:] - 0.5) ** 2 -
                                                   torch.cos(20 * torch.pi * (X[:, self.m - 1:] - 0.5)), dim=1,
                                                   keepdim=True))
        pop_con = torch.zeros((X.size(0), 2), device=X.device)
        pop_con[:, 0] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())
        pop_con[:, 1] = 0.5 - torch.exp(-g.squeeze())
        return f, pop_con

class DC2_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 10 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )
        g = 10 * (d - self.m + 1 + torch.sum((X[:, self.m - 1:] - 0.5) ** 2 -
                                                   torch.cos(20 * torch.pi * (X[:, self.m - 1:] - 0.5)), dim=1,
                                                   keepdim=True))

        pop_con = torch.zeros((X.size(0), 2), device=X.device)
        pop_con[:, 0] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())
        pop_con[:, 1] = 0.5 - torch.exp(-g.squeeze())
        return f, pop_con

class DC3_DTLZ1(DTLZ):
    def __init__(self, d: int = 7, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        m = self.m
        n, d = X.size()
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        flip_cumprod = torch.flip(
            torch.cumprod(
                torch.cat([torch.ones((n, 1), device=X.device), X[:, : m - 1]], dim=1),
                dim=1,
            ),
            dims=[1],
        )
        rest_part = torch.cat(
            [
                torch.ones((n, 1), device=X.device),
                1 - torch.flip(X[:, : m - 1], dims=[1]),
            ],
            dim=1,
        )
        f = 0.5 * (1 + g) * flip_cumprod * rest_part
        pop_con = torch.zeros((X.size(0), self.m), device=X.device)
        pop_con[:, :self.m - 1] = 0.5 - torch.cos(3 * torch.pi * X[:, :self.m - 1])
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1 :] - 0.5) ** 2 -
                                                    torch.cos(20 * torch.pi * (X[:, self.m - 1 :] - 0.5)), dim=1, keepdim=True))
        pop_con[:, self.m - 1] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())

        return f, pop_con

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            R = torch.zeros((100, 2), dtype=torch.float32)
            R[:, 0] = torch.linspace(0, 1, 100)
            R[:, 1] = 1 - R[:, 0]

            R[torch.cos(3 * torch.pi * R[:, 0]) < 0.5, :] = float('nan')
            R /= 2

            R = R[~torch.isnan(R).any(dim=1)]

        elif self.m == 3:
            a = torch.linspace(0, 1, 40).view(-1, 1)
            x = a @ a.t()
            y = a * (1 - a.t())
            z = (1 - a) * torch.ones_like(a.t())

            mask1 = torch.cos(3 * torch.pi * a @ torch.ones((1, 40))) < 0.5
            mask2 = torch.cos(3 * torch.pi * torch.ones((40, 1)) @ a.t()) < 0.5
            z[mask1 | mask2] = float('nan')

            R = torch.cat((x.view(-1,1) / 2, y.view(-1,1) / 2, z.view(-1,1) / 2), dim=1)
            R = R[~torch.isnan(R).any(dim=1)]

        else:
            return torch.tensor([])

        return R

class DC3_DTLZ3(DTLZ2):
    def __init__(self, d: int = 12, m: int = 3, ref_num: int = 1000):
        super().__init__(d, m, ref_num)

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        n, d = X.size()
        m = self.m
        g = 100 * (
            d
            - m
            + 1
            + torch.sum(
                (X[:, m - 1 :] - 0.5) ** 2 - torch.cos(20 * torch.pi * (X[:, m - 1 :] - 0.5)),
                dim=1,
                keepdim=True,
            )
        )
        f = (
            (1 + g)
            * torch.flip(
                torch.cumprod(
                    torch.cat(
                        [
                            torch.ones((n, 1), device=X.device),
                            torch.maximum(
                                torch.cos(X[:, : m - 1] * torch.pi / 2),
                                torch.tensor(0.0, device=X.device),
                            ),
                        ],
                        dim=1,
                    ),
                    dim=1,
                ),
                dims=[1],
            )
            * torch.cat(
                [
                    torch.ones((n, 1), device=X.device),
                    torch.sin(torch.flip(X[:, : m - 1], dims=[1]) * torch.pi / 2),
                ],
                dim=1,
            )
        )

        pop_con = torch.zeros((X.size(0), self.m), device=X.device)
        pop_con[:, :self.m - 1] = 0.5 - torch.cos(3 * torch.pi * X[:, :self.m - 1])
        g = 100 * (self.d - self.m + 1 + torch.sum((X[:, self.m - 1 :] - 0.5) ** 2 -
                                                    torch.cos(20 * torch.pi * (X[:, self.m - 1 :] - 0.5)), dim=1, keepdim=True))
        pop_con[:, self.m - 1] = 0.5 - torch.cos(3 * torch.pi * g.squeeze())

        return f, pop_con

    def pf(self) -> torch.Tensor:
        if self.m == 2:
            x = torch.linspace(0, torch.pi / 2, 100).view(-1, 1)
            R = torch.zeros((100, 2), dtype=torch.float32)
            R[:, 0] = torch.cos(x)
            R[:, 1] = torch.sin(x)
            R[torch.cos(6 * x) < 0.5, :] = float('nan')

        elif self.m == 3:
            a = torch.linspace(0, torch.pi / 2, 40).view(-1, 1)
            x = torch.cos(a) @ torch.cos(a.t())
            y = torch.cos(a) @ torch.sin(a.t())
            z = torch.sin(a) @ torch.ones((1, 40))


            mask1 = torch.cos(6 * a @ torch.ones((1, 40))) < 0.5
            mask2 = torch.cos(6 * torch.ones((40, 1)) @ a.t()) < 0.5
            z[mask1 | mask2] = float('nan')
            R = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), dim=1)
        else:
            return torch.tensor([])

        return R[~torch.isnan(R).any(dim=1)]