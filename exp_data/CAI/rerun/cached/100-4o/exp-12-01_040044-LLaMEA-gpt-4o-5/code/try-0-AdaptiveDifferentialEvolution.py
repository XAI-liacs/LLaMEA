import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # Population size
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Differential weight
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = x0 + self.f * (x1 - x2)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return pop[best_index], fitness[best_index]

# Example usage:
# optimizer = AdaptiveDifferentialEvolution(budget=10000, dim=10)
# best_solution, best_value = optimizer(some_black_box_function)