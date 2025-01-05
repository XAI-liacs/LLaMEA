import numpy as np

class HCS_AOBL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nest_size = min(30, budget // 2)
        self.pa = 0.25  # Abandonment rate
        self.best_nest = None
        self.best_value = np.inf
        self.step_size = 0.01  # Learning step size

    def levy_flight(self):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def opposite_sol(self, candidate, lb, ub):
        opposite = lb + (ub - candidate)
        return np.clip(opposite, lb, ub)

    def init_nests(self, lb, ub):
        return np.random.uniform(lb, ub, (self.nest_size, self.dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        nests = self.init_nests(lb, ub)
        nest_values = np.full(self.nest_size, np.inf)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.nest_size):
                if evaluations >= self.budget:
                    break
                
                candidate = nests[i] + self.step_size * self.levy_flight()
                candidate = np.clip(candidate, lb, ub)
                opposite_candidate = self.opposite_sol(candidate, lb, ub)

                candidate_value = func(candidate)
                opposite_value = func(opposite_candidate)
                evaluations += 2

                if candidate_value < nest_values[i]:
                    nest_values[i] = candidate_value
                    nests[i] = candidate
                if opposite_value < nest_values[i]:
                    nest_values[i] = opposite_value
                    nests[i] = opposite_candidate

                if candidate_value < self.best_value:
                    self.best_value = candidate_value
                    self.best_nest = candidate.copy()
                if opposite_value < self.best_value:
                    self.best_value = opposite_value
                    self.best_nest = opposite_candidate.copy()

            abandon = np.random.rand(self.nest_size) < self.pa
            for j in range(self.nest_size):
                if abandon[j]:
                    nests[j] = np.random.uniform(lb, ub, self.dim)

        return self.best_nest, self.best_value