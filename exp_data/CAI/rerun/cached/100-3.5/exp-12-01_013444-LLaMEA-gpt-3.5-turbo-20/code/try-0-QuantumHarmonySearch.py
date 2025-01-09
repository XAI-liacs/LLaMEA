import numpy as np

class QuantumHarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.3, band=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.band = band
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def _initialize_population(self, size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))
    
    def _quantum_gates(self, population):
        return np.where(np.random.rand(*population.shape) < 0.5, population, -population)
    
    def __call__(self, func):
        population = self._initialize_population(50)
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget - 50):
            new_member = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_member[i] = population[np.random.randint(len(population))][i]
                else:
                    new_member[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                    if np.random.rand() < self.par:
                        new_member[i] += self.band * self._quantum_gates(population[:, i].reshape(-1, 1))[0][0]
            new_fitness = func(new_member)
            
            if new_fitness < max(fitness):
                idx = np.argmax(fitness)
                population[idx] = new_member
                fitness[idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]