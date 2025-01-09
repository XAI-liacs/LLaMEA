import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, population_size=100, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation
                indices = list(range(0, i)) + list(range(i + 1, self.population_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                
                # Ensure mutation within bounds
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

                # Adaptive control parameters based on diversity
                # Change here: Using entropy-based diversity
                population_diversity = np.sum(-np.mean(np.log2(population + 1e-12), axis=0))  # Entropy-based diversity
                self.F = 0.4 + 0.2 * (population_diversity / (self.upper_bound - self.lower_bound))
                self.CR = 0.8 - 0.3 * (population_diversity / (self.upper_bound - self.lower_bound))
                self.F = np.clip(self.F, 0.1, 0.9)
                self.CR = np.clip(self.CR, 0.1, 1.0)

        return best