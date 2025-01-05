import numpy as np

class QuantumInspiredFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.alpha = 0.5  # Randomness reduction parameter
        self.beta0 = 1.0  # Initial attractiveness
        self.gamma = 1.0  # Light absorption coefficient
        self.quantum_factor_initial = 0.3
        self.quantum_factor_final = 0.1

    def quantum_attraction(self, firefly_i, firefly_j, brightness_i, brightness_j, eval_count):
        distance = np.linalg.norm(firefly_i - firefly_j)
        beta = self.beta0 * np.exp(-self.gamma * distance ** 2)
        lambda_factor = (eval_count / self.budget)
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        move = beta * (firefly_j - firefly_i) + quantum_factor * np.random.uniform(-1, 1, self.dim)
        return firefly_i + move

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        fireflies = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        brightness = np.array([func(firefly) for firefly in fireflies])
        
        eval_count = self.population_size
        global_best_index = np.argmin(brightness)
        global_best = fireflies[global_best_index]
        global_best_value = brightness[global_best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if brightness[j] < brightness[i]:
                        new_position = self.quantum_attraction(fireflies[i], fireflies[j], brightness[i], brightness[j], eval_count)
                        new_position = np.clip(new_position, bounds[:, 0], bounds[:, 1])
                        new_value = func(new_position)
                        eval_count += 1
                        if new_value < brightness[i]:
                            fireflies[i] = new_position
                            brightness[i] = new_value
                            if new_value < global_best_value:
                                global_best = new_position
                                global_best_value = new_value

                        if eval_count >= self.budget:
                            break
                if eval_count >= self.budget:
                    break

        return global_best