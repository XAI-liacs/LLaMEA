import numpy as np

class DifferentialEvolutionLevySA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def levy_flight(self, size, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        pop_size = 12 * self.dim  # Changed from 10 * dim
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]

        CR = 0.8  # Adjusted crossover probability
        F = np.full(pop_size, 0.7)  # Modified differential weight
        temperature = 1.0

        for generation in range(self.budget - pop_size):
            for i in range(pop_size):
                if np.random.rand() < 0.2:  # Introduce Levy flight with some probability
                    step = 0.01 * self.levy_flight(self.dim)
                    mutant = np.clip(pop[i] + step, func.bounds.lb, func.bounds.ub)
                else:
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F[i] * (b - c), func.bounds.lb, func.bounds.ub)
                
                CR = 0.6 + (0.4 * (1 - (generation / (self.budget * 0.6))))  # Adjusted dynamic CR

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / temperature):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best = trial

            temperature *= 0.97 - 0.03 * (generation / self.budget)  # Adjusted cooling schedule

        return best, best_fitness