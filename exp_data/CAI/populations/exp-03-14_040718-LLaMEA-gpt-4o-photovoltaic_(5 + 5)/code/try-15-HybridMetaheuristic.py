import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.temperature = 1.0
        self.cooling_rate = 0.99
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Differential Evolution Mutation
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutation_factor = 0.5 + 0.3 * (np.random.rand() - 0.5)  # Dynamically adjust mutation factor
                mutant = np.clip(population[a] + mutation_factor * (population[b] - population[c]), lb, ub)
                
                # Simulated Annealing Crossover
                trial = np.where(np.random.rand(self.dim) < self.temperature, mutant, population[i])
                
                # Evaluate new candidate
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection with diversity preservation
                if trial_fitness < fitness[i] or np.random.rand() < 0.1:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Cooling schedule for Simulated Annealing
            self.temperature *= self.cooling_rate
            # Adaptive population size adjustment
            if self.temperature < 0.5:
                self.population_size = min(int(self.population_size * 1.1), self.budget - evaluations)
        
        return best_solution