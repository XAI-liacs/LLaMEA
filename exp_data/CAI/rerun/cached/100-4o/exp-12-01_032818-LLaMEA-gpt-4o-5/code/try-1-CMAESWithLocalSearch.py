import numpy as np

class CMAESWithLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 4 + int(3 * np.log(dim))
        self.sigma = 0.3  # Initial step-size
        self.mean = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.cov = np.eye(dim)
        self.evaluations = 0

    def _clamp(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        best_solution = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            # Generate offspring from current mean and covariance
            offspring = np.random.multivariate_normal(self.mean, self.cov, self.population_size)
            offspring = np.array([self._clamp(individual) for individual in offspring])
            
            # Evaluate offspring
            values = np.array([func(ind) for ind in offspring])
            self.evaluations += self.population_size

            # Select the best individual
            indices = np.argsort(values)
            offspring = offspring[indices]
            values = values[indices]

            # Update best found solution
            if values[0] < best_value:
                best_value = values[0]
                best_solution = offspring[0]

            # Adapt mean and covariance matrix
            self.mean = np.mean(offspring[:self.population_size//2], axis=0)
            centered_offspring = offspring[:self.population_size//2] - self.mean
            self.cov = (1/self.population_size) * np.dot(centered_offspring.T, centered_offspring)

            # Perform a local search mutation on the best found solution
            for _ in range(3):  # Try 3 local mutations
                if self.evaluations >= self.budget:
                    break
                local_step = np.random.normal(0, self.sigma / 2, self.dim)
                local_candidate = self._clamp(best_solution + local_step)
                local_value = func(local_candidate)
                self.evaluations += 1
                if local_value < best_value:
                    best_value = local_value
                    best_solution = local_candidate
            
            # Dynamic population adjustment
            self.population_size = 4 + int(3 * np.log(self.dim)) + (self.budget - self.evaluations) // (10 * self.dim)

        return best_solution