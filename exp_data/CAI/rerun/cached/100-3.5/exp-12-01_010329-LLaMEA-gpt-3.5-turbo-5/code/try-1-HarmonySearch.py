import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def improve_solution(self, solution, hmcr=0.7, par=0.3):
        new_solution = np.copy(solution)
        for i in range(self.dim):
            if np.random.rand() < hmcr:
                new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound) if np.random.rand() < par else solution[i]
        return new_solution

    def optimize(self, func):
        harmonies = [self.generate_random_solution() for _ in range(self.budget)]
        for _ in range(self.budget):
            new_harmony = self.improve_solution(harmonies[np.random.randint(self.budget)])
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
                harmonies = sorted(harmonies, key=lambda x: func(x))
        return harmonies[0]