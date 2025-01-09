import numpy as np

class ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 5 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.strategy = 'rand/1/bin'
        
    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                
                # Selection
                trial_fit = func(trial)
                eval_count += 1
                
                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

            # Adaptation of parameters
            self.F = np.clip(np.random.normal(0.5, 0.3), 0.1, 0.9)
            self.CR = np.clip(np.random.normal(0.9, 0.1), 0.1, 1.0)
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]