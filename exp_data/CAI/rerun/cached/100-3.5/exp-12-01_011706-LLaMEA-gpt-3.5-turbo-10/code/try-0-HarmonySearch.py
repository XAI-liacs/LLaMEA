import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, harmony_memory_size=20, pitch_adjust_rate=0.1, pitch_adjust_bandwidth=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.pitch_adjust_bandwidth = pitch_adjust_bandwidth
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def pitch_adjust(solution):
            indices = np.random.choice(self.dim, int(self.pitch_adjust_rate * self.dim), replace=False)
            for idx in indices:
                solution[idx] += np.random.uniform(-self.pitch_adjust_bandwidth, self.pitch_adjust_bandwidth)
                solution[idx] = np.clip(solution[idx], self.lower_bound, self.upper_bound)
            return solution

        harmony_memory = [random_solution() for _ in range(self.harmony_memory_size)]
        harmony_memory_fitness = [func(h) for h in harmony_memory]
        
        for _ in range(self.budget - self.harmony_memory_size):
            new_solution = pitch_adjust(random_solution())
            new_fitness = func(new_solution)
            if new_fitness < max(harmony_memory_fitness):
                idx_replace = np.argmax(harmony_memory_fitness)
                harmony_memory[idx_replace] = new_solution
                harmony_memory_fitness[idx_replace] = new_fitness
        
        best_solution_idx = np.argmin(harmony_memory_fitness)
        return harmony_memory[best_solution_idx]