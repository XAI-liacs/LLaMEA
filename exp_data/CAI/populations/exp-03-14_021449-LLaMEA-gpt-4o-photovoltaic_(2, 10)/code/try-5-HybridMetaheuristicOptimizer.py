import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Reduced cooling rate for slower temperature decrease
        self.mutation_scale = 0.8
        
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Differential Evolution-like Mutation
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                self.mutation_scale = np.random.uniform(0.5, 1.0)  # Self-adaptive mutation scale
                mutant = x1 + self.mutation_scale * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)
                
                # Simulated Annealing-like Probabilistic Acceptance
                trial = mutant if np.random.rand() < 0.9 else population[i]  # Increased trial acceptance probability
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i] or np.exp((fitness[i] - trial_fitness) / self.temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
            
            self.temperature *= self.cooling_rate
        
        return best_solution