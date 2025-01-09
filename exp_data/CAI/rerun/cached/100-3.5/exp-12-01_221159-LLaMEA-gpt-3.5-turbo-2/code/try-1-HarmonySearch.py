class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (budget, dim))

    def __call__(self, func):
        for i in range(self.budget):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            for j in range(self.dim):
                if np.random.rand() < 0.7:
                    new_solution[j] = np.random.choice(self.harmony_memory[:, j])
            if func(new_solution) < func(self.harmony_memory[i]):
                self.harmony_memory[i] = new_solution
        return self.harmony_memory[np.argmin([func(sol) for sol in self.harmony_memory])]