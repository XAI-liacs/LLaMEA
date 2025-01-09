import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9
        
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.eval_count = 0
    
    def __call__(self, func):
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Select three random indices different from current index i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Generate mutant vector
                mutant_vector = self.population[a] + \
                                self.mutation_factor * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                # Crossover
                trial_vector = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial_vector[j] = mutant_vector[j]
                
                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                
                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
            
            # Adaptive strategy adjustment
            self._adapt_strategy()
        
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
    
    def _adapt_strategy(self):
        # Adapt mutation factor and crossover probability based on success rate
        success_rate = np.mean(self.fitness < np.median(self.fitness))  # Simple heuristic
        self.mutation_factor = 0.5 + 0.2 * (1 - success_rate)  # Adjust mutation factor
        self.crossover_prob = 0.9 - 0.2 * (1 - success_rate)  # Adjust crossover probability