import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = min(100, self.budget // (2 * self.dim))  # Adjusted population size
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx]
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Select three random indices different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation
                mutant = pop[a] + self.mutation_factor * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                # Evaluate trial vector
                f = func(trial)
                eval_count += 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

            # Adapt mutation factor and crossover probability based on fitness improvement rate
            improvement_rate = (np.min(fitness) - self.f_opt) / self.f_opt
            self.mutation_factor = 0.5 + (0.3 * improvement_rate)
            self.crossover_prob = 0.7 + (0.2 * (1 - improvement_rate))
            
            if eval_count >= self.budget:
                break
        
        return self.f_opt, self.x_opt