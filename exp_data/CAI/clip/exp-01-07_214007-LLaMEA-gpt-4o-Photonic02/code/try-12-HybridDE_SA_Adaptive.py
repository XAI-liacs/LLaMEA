import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cross_prob = 0.9
        self.initial_f_scale = 0.8
        self.temperature = 1.0
        self.cooling_rate = 0.95
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Adaptive Differential Evolution mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                # Adaptive mutation factor based on convergence progress
                f_scale = self.initial_f_scale + 0.2 * (1 - eval_count / self.budget)
                mutant = np.clip(a + f_scale * (b - c), lb, ub)
                
                self.cross_prob = 0.8 + 0.2 * (1 - eval_count / self.budget)
                crossover_mask = np.random.rand(self.dim) < self.cross_prob
                trial = np.where(crossover_mask, mutant, population[i])

                # Dynamic temperature annealing in Simulated Annealing
                trial_fitness = func(trial)
                eval_count += 1
                acceptance_probability = np.exp((fitness[i] - trial_fitness) / self.temperature)
                if trial_fitness < fitness[i] or np.random.rand() < acceptance_probability:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Cooling the temperature more gradually
            self.temperature *= self.cooling_rate

        return best_solution