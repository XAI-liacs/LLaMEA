import numpy as np

class BioInspiredFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, dim)
        self.alpha = 0.5  # Randomness parameter
        self.beta_min = 0.2  # Minimum attraction
        self.gamma = 1.0  # Light absorption coefficient
        self.adaptive_rate = 0.01  # Rate of adaptive adjustment of alpha

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        fireflies = np.random.uniform(lb, ub, (self.population_size, self.dim))
        light_intensity = np.array([func(fireflies[i]) for i in range(self.population_size)])
        evaluations = self.population_size
        best_index = np.argmin(light_intensity)
        best_firefly = fireflies[best_index].copy()

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if light_intensity[i] > light_intensity[j]:
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * distance**2)
                        movement = beta * (fireflies[j] - fireflies[i]) + self.alpha * (np.random.rand(self.dim) - 0.5) * (ub - lb)
                        fireflies[i] = np.clip(fireflies[i] + movement, lb, ub)

            light_intensity = np.array([func(fireflies[i]) for i in range(self.population_size)])
            evaluations += self.population_size

            current_best_index = np.argmin(light_intensity)
            if light_intensity[current_best_index] < light_intensity[best_index]:
                best_index = current_best_index
                best_firefly = fireflies[best_index].copy()

            # Adaptive randomness reduction
            self.alpha *= 1 - self.adaptive_rate * (evaluations / self.budget)

        return best_firefly, light_intensity[best_index]