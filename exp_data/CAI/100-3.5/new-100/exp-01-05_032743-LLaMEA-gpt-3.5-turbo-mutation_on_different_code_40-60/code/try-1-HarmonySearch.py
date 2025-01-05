import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01, bw_range=(0.01, 0.1)):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.bw_range = bw_range
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        harmony_memory = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.dim,))
        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[d]
                else:
                    new_solution[d] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    if np.random.rand() < self.par:
                        adj_bw = np.random.uniform(self.bw_range[0], self.bw_range[1])
                        new_solution[d] += np.random.uniform(-adj_bw, adj_bw)
                    new_solution[d] = np.clip(new_solution[d], func.bounds.lb, func.bounds.ub)

            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
                harmony_memory = np.vstack((harmony_memory, new_solution))
                harmony_memory = harmony_memory[harmony_memory[:, -1].argsort()][:self.dim]

        return self.f_opt, self.x_opt