import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim  # A common heuristic for population size
        self.mutation_factor = 0.5       # Initial mutation factor
        self.crossover_prob = 0.9        # Initial crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        decay_rate = 0.99  # Added decay rate for adaptive parameters
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), lb, ub)
                
                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                
            # Adaptation of parameters using decay
            self.mutation_factor = np.clip(self.mutation_factor * decay_rate + np.random.normal(0, 0.05), 0.1, 0.9)
            self.crossover_prob = np.clip(self.crossover_prob * decay_rate + np.random.normal(0, 0.05), 0.1, 1.0)
        
        best_index = np.argmin(fitness)
        self.f_opt = fitness[best_index]
        self.x_opt = population[best_index]

        return self.f_opt, self.x_opt