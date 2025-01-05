import numpy as np

class AdaptiveHarmonySearchDifferentialMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par_min = 0.2  # Initial Pitch Adjustment Rate
        self.par_max = 0.8  # Maximum Pitch Adjustment Rate
        self.f_min = 0.1  # Minimum differential mutation factor
        self.f_max = 0.9  # Maximum differential mutation factor
        self.position = None
        self.harmony_memory = None
        self.hm_scores = None
        self.best_harmony = None
        self.best_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.harmony_memory = np.copy(self.position)
        self.hm_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(h) for h in self.harmony_memory])
        for i in range(self.population_size):
            if scores[i] < self.hm_scores[i]:
                self.hm_scores[i] = scores[i]
                self.harmony_memory[i] = self.position[i]
            if scores[i] < self.best_score:
                self.best_score = scores[i]
                self.best_harmony = self.position[i]
        return scores

    def adapt_par(self, iteration, max_iterations):
        return self.par_min + (self.par_max - self.par_min) * (iteration / max_iterations)

    def adapt_f(self, iteration, max_iterations):
        return self.f_min + (self.f_max - self.f_min) * (1 - iteration / max_iterations)

    def differential_mutation(self, harmony, f):
        indexes = np.random.choice(self.population_size, 3, replace=False)
        mutant = self.harmony_memory[indexes[0]] + f * (self.harmony_memory[indexes[1]] - self.harmony_memory[indexes[2]])
        return np.clip(mutant, 0, 1)

    def generate_new_harmony(self, iteration, max_iterations):
        new_harmony = np.zeros(self.dim)
        par = self.adapt_par(iteration, max_iterations)
        f = self.adapt_f(iteration, max_iterations)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                indx = np.random.randint(self.population_size)
                new_harmony[i] = self.harmony_memory[indx][i]
                if np.random.rand() < par:
                    new_harmony[i] += np.random.uniform(-1, 1) * f
            else:
                mutant = self.differential_mutation(new_harmony, f)
                new_harmony[i] = mutant[i]
        return new_harmony

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        while func_calls < self.budget:
            for i in range(self.population_size):
                self.position[i] = self.generate_new_harmony(iteration, max_iterations)
            scores = self.evaluate(func)
            func_calls += self.population_size
            iteration += 1
        
        return self.best_harmony, self.best_score