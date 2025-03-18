import numpy as np

class HSBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 10
        bees_count = 5
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(sol) for sol in harmony_memory])
        secondary_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        secondary_memory_fitness = np.array([func(sol) for sol in secondary_memory])
        evals = 2 * harmony_memory_size
        
        while evals < self.budget:
            for bee in range(bees_count):
                if evals >= self.budget:
                    break
                
                if np.random.rand() < 0.5:
                    best_idx = np.argmin(harmony_memory_fitness)
                    new_solution = np.copy(harmony_memory[best_idx])
                else:
                    if np.random.rand() < 0.7:
                        new_solution = np.copy(harmony_memory[np.random.choice(harmony_memory_size)])
                    else:
                        new_solution = np.copy(secondary_memory[np.random.choice(harmony_memory_size)])
                
                pitch_adj_rate = 0.15 + (0.35 - 0.15) * (1 - evals / self.budget)
                for i in range(self.dim):
                    if np.random.rand() < pitch_adj_rate:
                        new_solution[i] += np.random.uniform(-0.3, 0.3) * (ub[i] - lb[i]) * 0.3
                
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                evals += 1

                worst_idx = np.argmax(harmony_memory_fitness)
                if new_fitness < harmony_memory_fitness[worst_idx]:
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness
                
                worst_secondary_idx = np.argmax(secondary_memory_fitness)
                if new_fitness < secondary_memory_fitness[worst_secondary_idx]:
                    secondary_memory[worst_secondary_idx] = new_solution
                    secondary_memory_fitness[worst_secondary_idx] = new_fitness

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]