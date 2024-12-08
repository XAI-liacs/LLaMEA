import numpy as np

class HybridQIPSO_DAENM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 8 * dim  # Reduced population size for faster convergence
        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, dim))  # Reduced initial velocity range
        self.personal_best = np.copy(self.pop)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.func_evals = 0
        self.alpha = 1.1  # Slightly increased reflection coefficient
        self.beta = 0.6  # Adjusted contraction coefficient
        self.gamma = 2.1  # Slightly increased expansion coefficient
        self.inertia_weight = 0.6  # Slightly decreased inertia weight
        self.cognitive_weight = 1.6  # Slightly increased cognitive weight
        self.social_weight = 1.4  # Slightly decreased social weight
        self.adaptive_rate = 0.05  # New adaptive rate for dynamic adjustment

    def update_particle(self, i, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                              self.cognitive_weight * r1 * (self.personal_best[i] - self.pop[i]) +
                              self.social_weight * r2 * (self.global_best - self.pop[i]))
        self.pop[i] = np.clip(self.pop[i] + self.velocities[i], self.lb, self.ub)
        f_value = func(self.pop[i])
        self.func_evals += 1
        if f_value < self.personal_best_fitness[i]:
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
        if f_value < self.global_best_fitness:
            self.global_best_fitness = f_value
            self.global_best = self.pop[i]
        # Dynamic adaptation based on performance
        self.inertia_weight *= (1 - self.adaptive_rate) if f_value > self.global_best_fitness else (1 + self.adaptive_rate)

    def __call__(self, func):
        for i in range(self.pop_size):
            f_value = func(self.pop[i])
            self.func_evals += 1
            self.personal_best_fitness[i] = f_value
            self.personal_best[i] = self.pop[i]
            if f_value < self.global_best_fitness:
                self.global_best_fitness = f_value
                self.global_best = self.pop[i]

        while self.func_evals < self.budget:
            for i in range(self.pop_size):
                if self.func_evals >= self.budget:
                    return self.global_best
                self.update_particle(i, func)

            if self.func_evals < self.budget:
                best_idx = np.argmin(self.personal_best_fitness)
                simplex = np.array([self.personal_best[best_idx]] + [self.personal_best[np.random.randint(self.pop_size)] for _ in range(self.dim)])
                simplex_fitness = np.array([func(p) for p in simplex])
                self.func_evals += len(simplex)

                while self.func_evals < self.budget:
                    order = np.argsort(simplex_fitness)
                    simplex = simplex[order]
                    simplex_fitness = simplex_fitness[order]

                    centroid = np.mean(simplex[:-1], axis=0)
                    reflected = np.clip(centroid + self.alpha * (centroid - simplex[-1]), self.lb, self.ub)
                    f_reflected = func(reflected)
                    self.func_evals += 1

                    if f_reflected < simplex_fitness[0]:
                        expanded = np.clip(centroid + self.gamma * (reflected - centroid), self.lb, self.ub)
                        f_expanded = func(expanded)
                        self.func_evals += 1

                        if f_expanded < f_reflected:
                            simplex[-1] = expanded
                            simplex_fitness[-1] = f_expanded
                        else:
                            simplex[-1] = reflected
                            simplex_fitness[-1] = f_reflected
                    elif f_reflected < simplex_fitness[-2]:
                        simplex[-1] = reflected
                        simplex_fitness[-1] = f_reflected
                    else:
                        contracted = np.clip(centroid + self.beta * (simplex[-1] - centroid), self.lb, self.ub)
                        f_contracted = func(contracted)
                        self.func_evals += 1

                        if f_contracted < simplex_fitness[-1]:
                            simplex[-1] = contracted
                            simplex_fitness[-1] = f_contracted
                        else:
                            for j in range(1, len(simplex)):
                                simplex[j] = np.clip(simplex[0] + 0.5 * (simplex[j] - simplex[0]), self.lb, self.ub)
                                simplex_fitness[j] = func(simplex[j])
                            self.func_evals += len(simplex) - 1

                best_simplex_idx = np.argmin(simplex_fitness)
                if simplex_fitness[best_simplex_idx] < self.personal_best_fitness[best_idx]:
                    self.personal_best[best_idx] = simplex[best_simplex_idx]
                    self.personal_best_fitness[best_idx] = simplex_fitness[best_simplex_idx]

        return self.global_best