import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        def initialize_harmony_memory(size):
            return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))
        
        def adjust_bandwidth(hm, bw):
            return hm + np.random.uniform(-bw, bw, hm.shape)
        
        def evaluate_solution(solution, func):
            return func(solution)
        
        def update_harmony_memory(hm, new_solution, new_fitness):
            idx = np.argmax(hm_fitness)
            if new_fitness < hm_fitness[idx]:
                hm[idx] = new_solution
                hm_fitness[idx] = new_fitness
        
        harmony_memory = initialize_harmony_memory(10)
        hm_fitness = np.array([evaluate_solution(solution, func) for solution in harmony_memory])
        bandwidth = 1.0
        
        for _ in range(self.budget - 10):
            new_solution = adjust_bandwidth(harmony_memory, bandwidth)
            new_fitness = evaluate_solution(new_solution, func)
            update_harmony_memory(harmony_memory, new_solution, new_fitness)
            bandwidth *= 0.9  # Decrease bandwidth over iterations
        
        best_idx = np.argmin(hm_fitness)
        return harmony_memory[best_idx]