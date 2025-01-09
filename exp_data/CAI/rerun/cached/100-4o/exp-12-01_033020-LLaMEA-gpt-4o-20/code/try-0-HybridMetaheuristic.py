import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Simulated Annealing-inspired Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Evaluate Trial Solution
                trial_fitness = func(trial)
                evaluations += 1
                
                # Acceptance Criterion
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / (1 + i)):
                    population[i] = trial
                    fitness[i] = trial_fitness
        
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]