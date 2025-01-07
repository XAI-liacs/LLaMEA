import numpy as np

class QGHS_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.harmony_memory_size = 30
        self.harmony_consideration_rate = 0.9
        self.pitch_adjustment_rate = 0.3
        self.mutation_rate = 0.1
        self.population = None
        self.harmony_memory = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.harmony_memory = np.copy(self.population)
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def quantum_genetic_encoding(self):
        return np.random.uniform(-1, 1, (self.population_size, self.dim))

    def harmony_search(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.harmony_consideration_rate:
                harmony_index = np.random.randint(self.harmony_memory_size)
                harmony_vector = self.harmony_memory[harmony_index]
                pitch_adjustment = np.random.uniform(-1, 1, self.dim) * self.pitch_adjustment_rate
                new_harmony = np.clip(harmony_vector + pitch_adjustment, lb, ub)
            else:
                new_harmony = np.random.uniform(lb, ub, self.dim)
            
            if np.random.rand() < self.mutation_rate:
                qg_encoding = self.quantum_genetic_encoding()
                new_harmony = np.clip(self.best_solution + qg_encoding[i] * np.abs(new_harmony - self.best_solution), lb, ub)

            trial_score = self.evaluate(new_harmony)
            if trial_score < self.scores[i]:
                self.population[i] = new_harmony
                self.scores[i] = trial_score
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_solution = new_harmony.copy()
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            self.harmony_search(lb, ub)
            self.harmony_memory = np.copy(self.population)

        return {'solution': self.best_solution, 'fitness': self.best_score}