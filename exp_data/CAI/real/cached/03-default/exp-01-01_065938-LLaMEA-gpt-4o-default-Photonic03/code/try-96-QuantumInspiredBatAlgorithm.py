import numpy as np

class QuantumInspiredBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.A = 0.5  # Loudness
        self.r_min, self.r_max = 0.0, 1.0  # Pulse rate bounds
        self.q_min, self.q_max = 0.0, 2.0  # Frequency bounds
        self.alpha = 0.9  # Loudness reduction factor
        self.gamma = 0.9  # Pulse rate increase factor
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.cos(np.pi * population_quantum)
        velocity = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size
        pulse_rate = np.random.uniform(self.r_min, self.r_max, self.population_size)
        frequency = np.random.uniform(self.q_min, self.q_max, self.population_size)

        while evaluations < self.budget:
            for i in range(self.population_size):
                beta = np.random.rand()
                frequency[i] = self.q_min + (self.q_max - self.q_min) * beta
                velocity[i] = velocity[i] + (pop[i] - best_global) * frequency[i]
                candidate = pop[i] + velocity[i]

                if np.random.rand() > pulse_rate[i]:
                    candidate = best_global + 0.001 * np.random.normal(size=self.dim)

                candidate = np.clip(candidate, lb, ub)
                candidate_fitness = func(candidate)
                evaluations += 1

                if candidate_fitness < fitness[i] and np.random.rand() < self.A:
                    pop[i] = candidate
                    fitness[i] = candidate_fitness
                    pulse_rate[i] = self.r_min + (self.r_max - self.r_min) * self.gamma
                    self.A *= self.alpha

                    if candidate_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = candidate

            self.history.append(best_global)

        return best_global