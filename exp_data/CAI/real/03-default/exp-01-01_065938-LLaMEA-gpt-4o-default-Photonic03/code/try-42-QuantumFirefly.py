import numpy as np

class QuantumFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # Randomness reduction factor
        self.gamma = 1.0  # Light absorption coefficient
        self.beta_min = 0.2  # Minimum attractiveness
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.sin(np.pi * population)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[i] > fitness[j]:
                        r = np.linalg.norm(pop[i] - pop[j])
                        beta = self.beta_min + (1 - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        
                        # Quantum-inspired update
                        noise = np.random.uniform(-0.5, 0.5, self.dim)
                        step = beta * (pop[j] - pop[i]) + self.alpha * noise * (ub - lb)
                        pop[i] += step
                        pop[i] = np.clip(pop[i], lb, ub)

                        current_fitness = func(pop[i])
                        evaluations += 1

                        if current_fitness < fitness[i]:
                            fitness[i] = current_fitness
                            if current_fitness < fitness[best_idx]:
                                best_idx = i
                                best_global = pop[i]

            self.alpha *= 0.99  # Reduce randomness
            self.history.append(best_global)

        return best_global