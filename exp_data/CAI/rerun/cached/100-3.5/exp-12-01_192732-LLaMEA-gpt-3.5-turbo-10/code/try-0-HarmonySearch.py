import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.4
        self.bandwidth = 0.01

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def adjust_parameters(iteration):
            self.hmcr = max(0.7 - iteration / self.budget, 0.1)
            self.par = min(0.4 + iteration / self.budget, 0.9)

        harmony_memory = initialize_harmony_memory()
        best_solution = harmony_memory[np.argmin([func(x) for x in harmony_memory])]
        
        for iteration in range(self.budget):
            adjust_parameters(iteration)
            new_solution = np.copy(harmony_memory[np.random.randint(self.budget)])
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    if np.random.rand() < self.par:
                        new_solution[i] = harmony_memory[np.random.randint(self.budget)][i]
                    else:
                        new_solution[i] = new_solution[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
            if func(new_solution) < func(harmony_memory[np.argmax([func(x) for x in harmony_memory])]):
                harmony_memory[np.argmax([func(x) for x in harmony_memory])] = new_solution
            if func(new_solution) < func(best_solution):
                best_solution = new_solution
                
        return best_solution