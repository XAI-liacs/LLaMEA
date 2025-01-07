import numpy as np

class FA_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5  # Randomness reduction coefficient
        self.beta0 = 1.0  # Attraction coefficient base value
        self.gamma = 1.0  # Absorption coefficient
        self.initial_temperature = 1000
        self.cooling_rate = 0.95

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize fireflies
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        intensity = np.array([func(p) for p in position])
        evaluations = self.population_size

        temperature = self.initial_temperature

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if intensity[i] > intensity[j]:
                        r = np.linalg.norm(position[i] - position[j]) / self.dim
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        position[i] += beta * (position[j] - position[i]) + self.alpha * (np.random.rand(self.dim) - 0.5)
                        position[i] = np.clip(position[i], lb, ub)

                        new_intensity = func(position[i])
                        evaluations += 1

                        if new_intensity < intensity[i]:
                            intensity[i] = new_intensity

                        if evaluations >= self.budget:
                            break

            # Simulated Annealing-like step
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                neighbor = position[i] + np.random.normal(0, self.alpha, self.dim)
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_intensity = func(neighbor)
                evaluations += 1

                delta = neighbor_intensity - intensity[i]
                if delta < 0 or np.exp(-delta / temperature) > np.random.rand():
                    position[i] = neighbor
                    intensity[i] = neighbor_intensity

            temperature *= self.cooling_rate

        best_index = np.argmin(intensity)
        return position[best_index], intensity[best_index]