import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(100, self.budget // 10)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.local_search_prob = 0.1
        
    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound,
                                (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= self.pop_size
        
        while self.budget > 0:
            for i in range(self.pop_size):
                # Differential Evolution Mutation
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = pop[a] + self.mutation_factor * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = np.copy(pop[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial[crossover_mask] = mutant[crossover_mask]
                
                # Evaluate trial
                trial_fitness = func(trial)
                self.budget -= 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                
                # Local Search
                if self.budget > 0 and np.random.rand() < self.local_search_prob:
                    local_trial = pop[i] + np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(local_trial, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_trial)
                    self.budget -= 1
                    if local_fitness < fitness[i]:
                        pop[i] = local_trial
                        fitness[i] = local_fitness
            
            # Re-evaluate population fitness if needed
            if self.budget > 0:
                fitness = np.array([func(ind) for ind in pop])
                self.budget -= self.pop_size
        
        # Return the best solution found
        best_index = np.argmin(fitness)
        return pop[best_index], fitness[best_index]