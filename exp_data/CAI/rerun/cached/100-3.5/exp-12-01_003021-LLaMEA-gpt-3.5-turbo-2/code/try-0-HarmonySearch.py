import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hmcr=0.7, par=0.6, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
    
    def generate_harmony(self):
        return np.random.uniform(-5.0, 5.0, self.dim)
    
    def adjust_pitch(self, harmony):
        for i in range(self.dim):
            if np.random.rand() < self.par:
                harmony[i] = harmony[i] + np.random.uniform(-self.bw, self.bw)
        return harmony

    def __call__(self, func):
        harmonies = [self.generate_harmony() for _ in range(self.budget)]
        best_solution = harmonies[0]
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_harmony = self.adjust_pitch(harmonies[np.random.choice(range(self.budget))])
            new_fitness = func(new_harmony)
            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness
        
        return best_solution