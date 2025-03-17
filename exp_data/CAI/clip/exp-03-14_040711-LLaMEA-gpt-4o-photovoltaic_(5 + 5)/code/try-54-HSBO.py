import numpy as np

class HSBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory_size = 12  # Increased harmony memory size
        bees_count = 7  # Increased bee count for better exploration
        harmony_memory = np.random.uniform(lb, ub, (harmony_memory_size, self.dim))
        harmony_memory_fitness = np.array([func(sol) for sol in harmony_memory])
        evals = harmony_memory_size
        
        while evals < self.budget:
            for bee in range(bees_count):
                if evals >= self.budget:
                    break
                
                # Harmony memory consideration
                new_solution = np.copy(harmony_memory[np.random.choice(harmony_memory_size)])
                
                # Adaptive pitch adjustment through dynamic local search
                pitch_adj_rate = 0.3 + 0.1 * (self.budget - evals) / self.budget  # Adjusted pitch adjustment rate
                for i in range(self.dim):
                    if np.random.rand() < pitch_adj_rate:
                        new_solution[i] += np.random.uniform(-0.5, 0.5) * (ub[i] - lb[i]) * 0.1  # Adjusted step size
                
                # Boundary check
                new_solution = np.clip(new_solution, lb, ub)
                
                # Evaluate new solution
                new_fitness = func(new_solution)
                evals += 1

                # Improved selective update strategy
                if new_fitness < np.percentile(harmony_memory_fitness, 50):  # Selectively update based on 50th percentile
                    worst_idx = np.argmax(harmony_memory_fitness)
                    harmony_memory[worst_idx] = new_solution
                    harmony_memory_fitness[worst_idx] = new_fitness

        best_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_idx]