import numpy as np
from scipy.optimize import minimize

class AdaptivePeriodicDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population_size = 10 + 3 * self.dim
        F, CR = 0.5, 0.9
        population = self.initialize_population(population_size)
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = np.copy(personal_best_positions[global_best_idx])
        eval_count = population_size

        while eval_count < self.budget:
            # Adaptive DE Mutation Scaling
            F = self.adaptive_mutation_scaling(eval_count, self.budget)
            
            for i in range(population_size):
                if eval_count >= self.budget:
                    break

                # DE Mutation and Crossover
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
        base_periodic_component = np.sin(np.linspace(0, 2 * np.pi, self.dim))
        population = np.random.uniform(self.lb, self.ub, (size, self.dim))
        for i in range(size):
            population[i] = 0.5 * (population[i] + base_periodic_component)
        return population

    def enforce_periodicity(self, vector):
        period = max(2, self.dim // 10)
        num_periods = self.dim // period
        for i in range(num_periods):
            mean_value = np.mean(vector[i*period:(i+1)*period])
            vector[i*period:(i+1)*period] = mean_value
        return vector

    def adaptive_mutation_scaling(self, eval_count, budget):
        return 0.9 - 0.8 * (eval_count / budget)