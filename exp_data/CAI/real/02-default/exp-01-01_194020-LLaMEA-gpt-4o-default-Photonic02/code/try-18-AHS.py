import numpy as np

class AHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.harmony_memory_size = 20
        self.harmony_memory = []
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par_min = 0.1  # Minimum Pitch Adjustment Rate
        self.par_max = 0.5  # Maximum Pitch Adjustment Rate

    def initialize_harmony_memory(self, lb, ub):
        for _ in range(self.harmony_memory_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            value = float('inf')
            self.harmony_memory.append({'position': position, 'value': value})

    def adaptive_pitch_adjustment(self, position, lb, ub, iteration, max_iter):
        par = self.par_min + ((self.par_max - self.par_min) * iteration / max_iter)
        if np.random.rand() < par:
            pitch_adjustment = (ub - lb) * (np.random.rand(self.dim) * 2 - 1) * 0.01
            position += pitch_adjustment
        return np.clip(position, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.initialize_harmony_memory(lb, ub)

        while evaluations < self.budget:
            new_position = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = self.harmony_memory[np.random.randint(self.harmony_memory_size)]
                    new_position[i] = selected_harmony['position'][i]
                else:
                    new_position[i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()
            
            new_position = self.adaptive_pitch_adjustment(new_position, lb, ub, evaluations, self.budget)
            new_value = func(new_position)
            evaluations += 1

            if new_value < self.best_value:
                self.best_value = new_value
                self.best_solution = new_position.copy()

            worst_index = np.argmax([h['value'] for h in self.harmony_memory])
            if new_value < self.harmony_memory[worst_index]['value']:
                self.harmony_memory[worst_index] = {'position': new_position, 'value': new_value}

        return self.best_solution, self.best_value