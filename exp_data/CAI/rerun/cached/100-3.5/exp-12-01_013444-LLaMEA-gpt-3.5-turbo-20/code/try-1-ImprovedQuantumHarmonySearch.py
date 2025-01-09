import numpy as np

class ImprovedQuantumHarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.3, band=0.01, de_weight=0.5, de_crossp=0.9):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.band = band
        self.de_weight = de_weight
        self.de_crossp = de_crossp
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def _initialize_population(self, size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))
    
    def _quantum_gates(self, population):
        return np.where(np.random.rand(*population.shape) < 0.5, population, -population)
    
    def _differential_evolution(self, population, f):
        mutant = population[np.random.choice(len(population), 3, replace=False)]
        v = population[np.argmin(f)] + self.de_weight * (mutant[0] - mutant[1])
        for i in range(self.dim):
            if np.random.rand() > self.de_crossp:
                v[i] = mutant[2][i]
            if v[i] < self.lower_bound:
                v[i] = self.lower_bound
            elif v[i] > self.upper_bound:
                v[i] = self.upper_bound
        return v
    
    def __call__(self, func):
        population = self._initialize_population(50)
        fitness = [func(ind) for ind in population]
        
        for _ in range(self.budget - 50):
            new_member = np.zeros(self.dim)
            
            if np.random.rand() < self.hmcr:
                new_member = population[np.random.randint(len(population))]
            else:
                new_member = self._differential_evolution(population, fitness)
                
            new_member += np.where(np.random.rand(self.dim) < self.par, self.band * self._quantum_gates(population), 0)
            new_fitness = func(new_member)
            
            if new_fitness < max(fitness):
                idx = np.argmax(fitness)
                population[idx] = new_member
                fitness[idx] = new_fitness
        
        best_idx = np.argmin(fitness)
        return population[best_idx]