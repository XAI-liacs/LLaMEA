import numpy as np

class AdaptiveQIGWO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.alpha_pos = np.zeros(dim)
        self.beta_pos = np.zeros(dim)
        self.delta_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        self.beta_score = float('inf')
        self.delta_score = float('inf')
        self.evaluations = 0
        self.phi = np.log(2)
        self.beta = 0.3
        self.leaders = {'alpha': None, 'beta': None, 'delta': None}

    def levy_flight(self, scale=0.01):
        u = np.random.normal(0, 1, self.dim) * scale
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step

    def _update_position(self, idx, func):
        a = 2 - self.evaluations * (2 / self.budget)
        A1 = 2 * a * np.random.rand(self.dim) - a
        A2 = 2 * a * np.random.rand(self.dim) - a
        A3 = 2 * a * np.random.rand(self.dim) - a
        C1 = 2 * np.random.rand(self.dim)
        C2 = 2 * np.random.rand(self.dim)
        C3 = 2 * np.random.rand(self.dim)

        D_alpha = np.abs(C1 * self.alpha_pos - self.positions[idx])
        D_beta = np.abs(C2 * self.beta_pos - self.positions[idx])
        D_delta = np.abs(C3 * self.delta_pos - self.positions[idx])

        X1 = self.alpha_pos - A1 * D_alpha
        X2 = self.beta_pos - A2 * D_beta
        X3 = self.delta_pos - A3 * D_delta

        new_pos = (X1 + X2 + X3) / 3

        if np.random.rand() < self.beta:
            new_pos += self.levy_flight()

        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        score = func(new_pos)
        self.evaluations += 1

        if score < self.alpha_score:
            self.alpha_score, self.alpha_pos = score, new_pos
        elif score < self.beta_score:
            self.beta_score, self.beta_pos = score, new_pos
        elif score < self.delta_score:
            self.delta_score, self.delta_pos = score, new_pos

        self.positions[idx] = new_pos

    def _adapt_leaders(self):
        if self.evaluations % (self.budget // 5) == 0:
            current_leader_scores = [self.alpha_score, self.beta_score, self.delta_score]
            sorted_indices = np.argsort(current_leader_scores)
            self.leaders['alpha'], self.leaders['beta'], self.leaders['delta'] = (
                self.positions[sorted_indices[0]], 
                self.positions[sorted_indices[1]], 
                self.positions[sorted_indices[2]]
            )
            self.alpha_pos, self.beta_pos, self.delta_pos = (
                self.leaders['alpha'],
                self.leaders['beta'],
                self.leaders['delta']
            )

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            if score < self.alpha_score:
                self.alpha_score, self.alpha_pos = score, self.positions[i]
            elif score < self.beta_score:
                self.beta_score, self.beta_pos = score, self.positions[i]
            elif score < self.delta_score:
                self.delta_score, self.delta_pos = score, self.positions[i]

            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.alpha_pos

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_position(i, func)
                if self.evaluations >= self.budget:
                    break
            self._adapt_leaders()

        return self.alpha_pos