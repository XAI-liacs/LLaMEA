import numpy as np

class QIPSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.population = None
        self.velocities = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.personal_best = None
        self.personal_best_scores = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.personal_best = self.population.copy()
        self.personal_best_scores = self.scores.copy()
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def evaluate(self, solution):
        return self.func(solution)

    def update_particle(self, idx, lb, ub):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        # Quantum inspired amplitude modulation
        amplitude = np.sin(self.evaluations / self.budget * np.pi)
        new_velocity = (self.velocities[idx] 
                        + amplitude * r1 * (self.personal_best[idx] - self.population[idx])
                        + amplitude * r2 * (self.best_solution - self.population[idx]))
        self.velocities[idx] = new_velocity
        new_position = self.population[idx] + new_velocity
        self.population[idx] = np.clip(new_position, lb, ub)

    def update_personal_best(self, idx):
        score = self.evaluate(self.population[idx])
        if score < self.personal_best_scores[idx]:
            self.personal_best[idx] = self.population[idx].copy()
            self.personal_best_scores[idx] = score
        if score < self.best_score:
            self.best_score = score
            self.best_solution = self.population[idx].copy()

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self.update_particle(i, lb, ub)
                self.update_personal_best(i)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}