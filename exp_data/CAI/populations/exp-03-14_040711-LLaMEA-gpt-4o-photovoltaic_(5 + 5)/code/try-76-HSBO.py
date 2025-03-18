import numpy as np

class HSBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 12
        bees_count = 7
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(sol) for sol in harmony_memory])
        evals = harmony_memory_size
        
        while evals < self.budget:
            fitness_variance = np.var(harmony_memory_fitness)
            dynamic_bees_count = max(3, int(bees_count * (1 + fitness_variance)))  # Dynamic bee count
            for bee in range(dynamic_bees_count):
                if evals >= self.budget:
                    break
                
                new_solution = np.copy(harmony_memory[np.random.choice(harmony_memory_size)])
                
                pitch_adj_rate = 0.3 + 0.1 * (self.budget - evals) / self.budget
                for i in range(self.dim):
                    if np.random.rand() < pitch_adj_rate:
                        new_solution[i] += np.random.uniform(-0.6, 0.6) * (ub[i] - lb[i]) * 0.1
                
                new_solution = np.clip(new_solution, lb, ub)
                
                new_fitness = func(new_solution)
                evals += 1

                if new_fitness < np.percentile(harmony_memory_fitness, 50):  # Update based on 50th percentile
                    worst_idx = np.argmax(harmony_memory_fitness)
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]