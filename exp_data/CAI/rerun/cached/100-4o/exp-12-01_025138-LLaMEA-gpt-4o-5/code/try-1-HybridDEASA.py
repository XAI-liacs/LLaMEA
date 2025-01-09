import numpy as np

class HybridDEASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for Differential Evolution
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.evaluations = 0

    def _differential_evolution(self, target_idx):
        a, b, c = self.population[np.random.choice(self.population_size, 3, replace=False)]
        F = 0.8  # Mutation factor (changed from random uniform to a fixed value)
        mutant = np.clip(a + F * (b - c), -5.0, 5.0)
        
        crossover_rate = np.random.uniform(0.1, 0.9)
        crossover = np.random.rand(self.dim) < crossover_rate
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True

        trial = np.where(crossover, mutant, self.population[target_idx])
        return trial

    def _adaptive_simulated_annealing(self, trial, current_fitness):
        T = max(1.0, 0.1 * (self.budget - self.evaluations) / self.budget * 100)  # Adjusted temperature schedule
        trial_fitness = func(trial)
        acceptance_probability = np.exp((current_fitness - trial_fitness) / T)
        
        if trial_fitness < self.best_fitness:
            self.best_solution = trial
            self.best_fitness = trial_fitness
        
        return trial_fitness, acceptance_probability
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                trial = self._differential_evolution(i)
                current_fitness = func(self.population[i])
                self.evaluations += 1
                
                trial_fitness, acceptance_probability = self._adaptive_simulated_annealing(trial, current_fitness)
                self.evaluations += 1

                if trial_fitness < current_fitness or np.random.rand() < acceptance_probability:
                    self.population[i] = trial
            
            if self.evaluations >= self.budget:
                break
        
        return self.best_solution, self.best_fitness