import numpy as np

class ACO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_ants = min(30, budget // 10)
        self.pheromone = np.ones((dim, 2))
        self.alpha = 1.0
        self.beta = 2.0
        self.evaporation_rate = 0.5
        self.f = 0.5  # DE mutation factor
        self.cr = 0.7  # DE crossover probability
        self.best_solution = None
        self.best_score = float('inf')

    def _select_route(self, lower_bound, upper_bound):
        route = np.zeros(self.dim)
        for i in range(self.dim):
            norm_pheromone = self.pheromone[i] / np.sum(self.pheromone[i])
            if np.random.rand() < norm_pheromone[0]:
                route[i] = lower_bound[i] + np.random.rand() * (upper_bound[i] - lower_bound[i])
            else:
                route[i] = upper_bound[i] - np.random.rand() * (upper_bound[i] - lower_bound[i])
        return route

    def _apply_de(self, population, lb, ub):
        new_population = np.copy(population)
        for i in range(self.num_ants):
            idxs = [idx for idx in range(self.num_ants) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            new_population[i] = np.where(cross_points, mutant, population[i])
        return new_population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.array([self._select_route(lb, ub) for _ in range(self.num_ants)])
        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(ind) for ind in population])
            evaluations += self.num_ants

            best_ant_idx = np.argmin(scores)
            if scores[best_ant_idx] < self.best_score:
                self.best_score = scores[best_ant_idx]
                self.best_solution = population[best_ant_idx]

            # Update pheromone trails
            for i in range(self.dim):
                if np.random.rand() < self.evaporation_rate:
                    self.pheromone[i] *= (1 - self.evaporation_rate)
                    self.pheromone[i][0] += self.alpha / (1.0 + scores[best_ant_idx])
                    self.pheromone[i][1] += self.beta / (1.0 + self.best_score)

            # Apply Differential Evolution to refine solutions
            population = self._apply_de(population, lb, ub)

        return self.best_solution