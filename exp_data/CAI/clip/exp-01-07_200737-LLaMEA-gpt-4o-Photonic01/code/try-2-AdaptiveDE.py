import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_agents = min(30, budget//10)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.agents = np.random.rand(self.num_agents, self.dim)
        self.best_agent = None
        self.best_score = float('inf')
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.agents = lb + (ub - lb) * np.random.rand(self.num_agents, self.dim)
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.num_agents):
                indices = list(range(self.num_agents))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = np.clip(self.agents[a] + self.mutation_factor * (self.agents[b] - self.agents[c]), lb, ub)
                trial_vector = np.copy(self.agents[i])
                
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_prob:
                        trial_vector[j] = mutant_vector[j]
                
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < func(self.agents[i]):
                    self.agents[i] = trial_vector
                
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_agent = trial_vector
            
            # Adaptive mutation and crossover
            if evaluations < self.budget/2:
                self.mutation_factor = np.random.uniform(0.5, 1.0)
                self.crossover_prob = np.random.uniform(0.6, 0.9)
            else:
                self.mutation_factor = np.random.uniform(0.4, 0.9)
                self.crossover_prob = np.random.uniform(0.5, 0.8)
        
        return self.best_agent