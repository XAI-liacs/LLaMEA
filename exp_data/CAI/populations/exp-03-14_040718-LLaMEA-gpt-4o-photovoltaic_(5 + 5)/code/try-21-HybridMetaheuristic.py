import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.initial_temperature = 1.0
        self.cooling_rate = 0.95  # Adjusted cooling rate for slower convergence
        self.temperature = self.initial_temperature
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Lévy flight Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                beta = 1.5  # Lévy flight exponent
                scale = (np.random.rand(self.dim) ** (-1.0 / beta))
                mutation_step = scale * (population[b] - population[c])
                mutant = np.clip(population[a] + mutation_step, lb, ub)
                
                # Crossover with adaptive threshold
                trial = np.where(np.random.rand(self.dim) < self.temperature, mutant, population[i])
                
                # Evaluate new candidate
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Adaptive cooling based on improvement
            self.temperature = self.initial_temperature * (0.99 ** (evaluations / self.population_size))
        
        return best_solution