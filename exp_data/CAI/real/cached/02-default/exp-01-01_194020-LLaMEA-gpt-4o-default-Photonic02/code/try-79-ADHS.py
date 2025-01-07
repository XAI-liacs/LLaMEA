import numpy as np

class ADHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 20
        self.harmonies = []
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_harmony_memory(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.harmony_memory_size)]

    def differential_mutation(self, harmony, target_index, lb, ub):
        indices = list(range(self.harmony_memory_size))
        indices.remove(target_index)
        r1, r2, r3 = np.random.choice(indices, 3, replace=False)
        
        mutation_vector = self.harmonies[r1] + 0.9 * (self.harmonies[r2] - self.harmonies[r3])
        mutation_vector = np.clip(mutation_vector, lb, ub)
        
        return mutation_vector

    def harmony_search_strategy(self, new_harmony, lb, ub):
        for i in range(self.dim):
            if np.random.rand() < 0.01:  # Harmony consideration rate
                random_index = np.random.randint(self.harmony_memory_size)
                new_harmony[i] = self.harmonies[random_index][i]
            elif np.random.rand() < 0.3:  # Pitch adjustment rate
                new_harmony[i] += (ub[i] - lb[i]) * (np.random.rand() - 0.5) * 0.2
        new_harmony = np.clip(new_harmony, lb, ub)
        return new_harmony

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.harmonies = self.initialize_harmony_memory(lb, ub)
        
        while evaluations < self.budget:
            for harmony_index in range(self.harmony_memory_size):
                if evaluations >= self.budget:
                    break
                
                new_harmony = self.differential_mutation(self.harmonies[harmony_index], harmony_index, lb, ub)
                new_harmony = self.harmony_search_strategy(new_harmony, lb, ub)
                
                value = func(new_harmony)
                evaluations += 1
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = new_harmony.copy()

                if value < func(self.harmonies[harmony_index]):
                    self.harmonies[harmony_index] = new_harmony

        return self.best_solution, self.best_value