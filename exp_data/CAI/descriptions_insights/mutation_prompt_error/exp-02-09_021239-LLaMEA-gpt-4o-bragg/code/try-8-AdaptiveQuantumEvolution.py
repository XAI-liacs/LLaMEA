import numpy as np

class AdaptiveQuantumEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.quantum_population = None
        self.bounds = None
        self.phi = 0.5 * np.pi  # Initial phase angle for quantum superposition
    
    def initialize_population(self):
        self.quantum_population = np.random.uniform(-1, 1, (self.population_size, self.dim))
    
    def measure_population(self):
        # Convert quantum states to real space solutions
        real_population = self.bounds.lb + ((self.bounds.ub - self.bounds.lb) * (np.sin(self.phi * self.quantum_population) + 1) / 2)
        return real_population
    
    def quantum_rotation(self, target_idx, best_idx):
        # Rotate quantum states towards the global best
        self.quantum_population[target_idx] = (
            self.quantum_population[target_idx]
            + np.random.uniform(0.05, 0.1) * (self.quantum_population[best_idx] - self.quantum_population[target_idx])
        )
    
    def adaptive_strategy(self, success_rate):
        # Adjust quantum phase angle based on success rate
        self.phi = 0.5 * np.pi * (1 - success_rate)
    
    def __call__(self, func):
        self.bounds = func.bounds
        self.initialize_population()
        real_population = self.measure_population()
        fitness = np.array([func(ind) for ind in real_population])
        remaining_budget = self.budget - self.population_size
        
        while remaining_budget > 0:
            best_idx = np.argmin(fitness)
            success_count = 0
            
            for i in range(self.population_size):
                if i != best_idx:
                    self.quantum_rotation(i, best_idx)
                    
                real_population = self.measure_population()
                trial_fitness = func(real_population[i])
                remaining_budget -= 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    success_count += 1
                
                if remaining_budget <= 0:
                    break
            
            success_rate = success_count / self.population_size
            self.adaptive_strategy(success_rate)
        
        best_idx = np.argmin(fitness)
        return real_population[best_idx]