import numpy as np
from scipy.optimize import minimize

class PSO_DE_HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None
    
    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population_size = 10 + 3 * self.dim
        F, CR = 0.5, 0.9
        w_max, w_min = 0.9, 0.4  # Adaptive inertia weight
        c1, c2 = 1.5, 1.5  # PSO parameters
        population = self.initialize_population(population_size)
        velocity = np.zeros((population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = np.copy(personal_best_positions[global_best_idx])
        eval_count = population_size

        while eval_count < self.budget:
            w = w_max - (w_max - w_min) * (eval_count / self.budget)
            c1 = 2.5 / (1 + np.exp(-10 * (eval_count / self.budget - 0.5)))
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (w * velocity[i] +
                               c1 * r1 * (personal_best_positions[i] - population[i]) +
                               c2 * r2 * (global_best_position - population[i]))

                population[i] = np.clip(population[i] + velocity[i], self.lb, self.ub)
                population[i] = self.enforce_periodicity(population[i], i)

                fitness = func(population[i])
                eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = personal_best_positions[i]

            F = 0.6 + 0.4 * np.sin(np.pi * eval_count / self.budget)
            CR = 0.9 - 0.5 * (eval_count / self.budget)
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = personal_best_positions[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                mutant = self.enforce_periodicity(mutant, i)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = personal_best_positions[i]

            if eval_count < self.budget:
                result = minimize(func, global_best_position, bounds=list(zip(self.lb, self.ub)), method='L-BFGS-B')
                eval_count += result.nfev
                if result.fun < personal_best_fitness[global_best_idx]:
                    global_best_position = result.x
                    personal_best_fitness[global_best_idx] = result.fun
                    personal_best_positions[global_best_idx] = global_best_position

        return global_best_position

    def initialize_population(self, size):
        return np.random.uniform(self.lb, self.ub, (size, self.dim))

    def enforce_periodicity(self, vector, index):
        period = 2 + (index % 2)
        num_periods = self.dim // period
        mean_value = np.mean(vector[:period])
        for i in range(num_periods):
            vector[i*period:(i+1)*period] = mean_value
        return vector