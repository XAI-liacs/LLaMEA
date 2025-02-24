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
        F_min, F_max = 0.3, 0.8  # Adaptive DE parameter F
        CR_min, CR_max = 0.7, 1.0  # Adaptive DE parameter CR
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
            # PSO Update
            w = w_max - (w_max - w_min) * (eval_count / self.budget)  # Adaptive inertia weight
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (w * velocity[i] +
                               c1 * r1 * (personal_best_positions[i] - population[i]) +
                               c2 * r2 * (global_best_position - population[i]))
                
                levy_step = self.levy_flight(self.dim)
                population[i] = np.clip(population[i] + velocity[i] + levy_step, self.lb, self.ub)
                population[i] = self.enforce_periodicity(population[i])

                fitness = func(population[i])
                eval_count += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = personal_best_positions[i]

            # DE Mutation and Crossover
            F = F_min + (F_max - F_min) * np.random.rand()
            CR = CR_min + (CR_max - CR_min) * np.random.rand()
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = personal_best_positions[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                mutant = self.enforce_periodicity(mutant)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best_position = personal_best_positions[i]

            # Local optimization with L-BFGS-B
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

    def enforce_periodicity(self, vector):
        period = 2
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector

    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        step = u / abs(v) ** (1 / beta)
        return 0.01 * step * np.random.rand(dim)