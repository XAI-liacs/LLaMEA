import numpy as np

class HybridHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.HM_size = 10
        self.HMCR = 0.9
        self.PAR_min = 0.1
        self.PAR_max = 0.5
        self.bw = 0.01 * (self.upper_bound - self.lower_bound)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.HM_size, dim))
        self.fitness = np.full(self.HM_size, np.inf)
    
    def __call__(self, func):
        eval_count = 0
        
        # Initialize harmony memory with initial evaluation
        for i in range(self.HM_size):
            self.fitness[i] = func(self.population[i])
            eval_count += 1
            if eval_count >= self.budget:
                return self.best_solution()
        
        while eval_count < self.budget:
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.HMCR:
                    idx = np.random.randint(self.HM_size)
                    new_harmony[j] = self.population[idx, j]
                    if np.random.rand() < self.PAR_min + (self.PAR_max - self.PAR_min) * eval_count / self.budget:
                        new_harmony[j] += np.random.uniform(-1, 1) * self.bw
                else:
                    new_harmony[j] = np.random.uniform(self.lower_bound, self.upper_bound)
            
            # Differential Evolution Crossover
            if np.random.rand() < 0.2:  # 20% chance to apply DE
                indices = np.random.choice(self.HM_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                F = 0.8
                new_harmony = x1 + F * (x2 - x3)
                new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)

            new_fitness = func(new_harmony)
            eval_count += 1

            if new_fitness < np.max(self.fitness):
                worst_idx = np.argmax(self.fitness)
                self.population[worst_idx] = new_harmony
                self.fitness[worst_idx] = new_fitness
        
        return self.best_solution()

    def best_solution(self):
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]