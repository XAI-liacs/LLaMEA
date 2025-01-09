import numpy as np

class HybridDifferentialEvolutionLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for DE
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.bounds = (-5.0, 5.0)
    
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= self.population_size

        while self.budget > 0:
            for i in range(self.population_size):
                if self.budget <= 0:
                    break
                
                # Differential Evolution Mutation and Crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, *self.bounds)
                
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate trial vector
                trial_fitness = func(trial)
                self.budget -= 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Local Search using a simple gradient approximation
                if self.budget > 0:
                    local_trial = trial + np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(local_trial, *self.bounds)
                    local_fitness = func(local_trial)
                    self.budget -= 1
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness

        best_index = np.argmin(fitness)
        return pop[best_index], fitness[best_index]