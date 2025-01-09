import numpy as np

class HarmonySearch:
    def __init__(self, budget=10000, dim=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim) for _ in range(len(func.bounds.lb) * 10)]
        return harmony_memory

    def update_harmony_memory(self, harmony_memory, new_solution, func):
        harmony_memory = harmony_memory[1:]
        harmony_memory.append(new_solution)
        harmony_memory.sort(key=lambda x: func(x))
        return harmony_memory

    def generate_new_solution(self, harmony_memory, func):
        new_solution = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                new_solution[i] = harmony_memory[np.random.randint(len(harmony_memory))][i]
                if np.random.rand() < self.par:
                    new_solution[i] = new_solution[i] + np.random.uniform(-self.bw, self.bw)
                    new_solution[i] = np.clip(new_solution[i], func.bounds.lb, func.bounds.ub)
            else:
                new_solution[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
        return new_solution

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory(func)
        
        for i in range(self.budget):
            new_solution = self.generate_new_solution(harmony_memory, func)
            f = func(new_solution)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_solution
            harmony_memory = self.update_harmony_memory(harmony_memory, new_solution, func)

        return self.f_opt, self.x_opt