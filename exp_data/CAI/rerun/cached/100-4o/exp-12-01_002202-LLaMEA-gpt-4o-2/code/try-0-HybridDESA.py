import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.temperature = 1000  # SA initial temperature
        self.cooling_rate = 0.95  # SA cooling rate
    
    def __call__(self, func):
        remaining_budget = self.budget
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        remaining_budget -= self.pop_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while remaining_budget > 0:
            # Differential Evolution Process
            for i in range(self.pop_size):
                if remaining_budget <= 0:
                    break
                a, b, c = population[np.random.choice(range(self.pop_size), 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                trial_fitness = func(trial)
                remaining_budget -= 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
            
            # Simulated Annealing Process
            for i in range(self.pop_size):
                if remaining_budget <= 0:
                    break
                candidate = population[i] + np.random.normal(0, 1, self.dim)
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                remaining_budget -= 1
                delta_e = candidate_fitness - fitness[i]
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / self.temperature):
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness
            
            # Cooling down the temperature
            self.temperature *= self.cooling_rate
        
        return best_solution, best_fitness