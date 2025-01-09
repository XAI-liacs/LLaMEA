import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7
        self.temperature = 100.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.crossover_probability
                trial = np.where(crossover, mutant, population[i])
                
                # Evaluate trial
                trial_fitness = func(trial)
                eval_count += 1
                
                # Simulated Annealing acceptance
                delta = trial_fitness - fitness[i]
                if delta < 0 or np.random.rand() < np.exp(-delta / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                
                # Cooling down the temperature
                self.temperature *= self.cooling_rate
                
                if eval_count >= self.budget:
                    break

        return best_solution