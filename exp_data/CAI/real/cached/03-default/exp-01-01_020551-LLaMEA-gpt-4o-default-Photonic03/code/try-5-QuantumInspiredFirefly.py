import numpy as np

class QuantumInspiredFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, dim)
        self.alpha = 0.5  # Randomization parameter
        self.gamma = 1.0  # Absorption coefficient
        self.beta0 = 1.0  # Initial attractiveness
        self.q_prob = 0.8 # Quantum behavior probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        brightness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if brightness[i] > brightness[j]:
                        dist = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * dist ** 2)
                        step = beta * (population[j] - population[i])
                        if np.random.rand() < self.q_prob:
                            # Quantum-inspired behavior
                            step += self.alpha * (np.random.rand(self.dim) - 0.5) * (ub - lb)
                        new_solution = population[i] + step
                        new_solution = np.clip(new_solution, lb, ub)
                        new_brightness = func(new_solution)
                        evaluations += 1
                        if new_brightness < brightness[i]:
                            population[i] = new_solution
                            brightness[i] = new_brightness
                            if evaluations >= self.budget:
                                break
                if evaluations >= self.budget:
                    break

        best_index = np.argmin(brightness)
        return population[best_index], brightness[best_index]