import numpy as np

class FireflyAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = max(2, budget // 1000)  # At least 2 fireflies
        self.alpha = 0.2  # Randomness parameter
        self.gamma = 1.0  # Absorption coefficient
        self.elitism_rate = 0.1  # Top fireflies to retain

    def __call__(self, func):
        # Initialize fireflies
        fireflies = np.random.uniform(func.bounds.lb, func.bounds.ub,
                                      (self.population_size, self.dim))
        fitness = np.array([func(ff) for ff in fireflies])

        eval_count = self.population_size

        while eval_count < self.budget:
            sorted_indices = np.argsort(fitness)
            fireflies = fireflies[sorted_indices]
            fitness = fitness[sorted_indices]

            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:  # Move firefly i towards j
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = np.exp(-self.gamma * r ** 2)

                        # Update position with adaptive randomness
                        step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + step * (1 - beta)

                        # Check bounds
                        fireflies[i] = np.clip(fireflies[i], func.bounds.lb, func.bounds.ub)

                        # Evaluate new position
                        new_fitness = func(fireflies[i])
                        eval_count += 1
                        if eval_count >= self.budget:
                            break

                        # Update if better
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness
                            if new_fitness < self.f_opt:
                                self.f_opt = new_fitness
                                self.x_opt = fireflies[i]
                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt