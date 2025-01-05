import numpy as np

class QTE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.tunnel_probability = 0.1
        self.niches = 5
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()
    
    def evaluate(self, solution):
        return self.func(solution)

    def mutate(self, target_idx):
        niche_size = self.population_size // self.niches
        niche_idx = (target_idx // niche_size) * niche_size
        indices = list(range(niche_idx, niche_idx + niche_size))
        indices.remove(target_idx)
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        F = np.random.uniform(0.5, 0.9)
        mutant = a + F * (b - c)
        return mutant

    def crossover(self, target, mutant):
        CR = np.random.uniform(0.2, 0.8)
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def quantum_tunnel(self):
        for i in range(self.population_size):
            if np.random.rand() < self.tunnel_probability:
                target = self.population[i]
                distance = np.linalg.norm(self.best_solution - target)
                tunnel_shift = np.random.uniform(-1, 1, self.dim) * distance
                tunnel_solution = target + tunnel_shift
                tunnel_solution = np.clip(tunnel_solution, self.func.bounds.lb, self.func.bounds.ub)
                self.select(i, tunnel_solution)

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self.quantum_tunnel()

        return {'solution': self.best_solution, 'fitness': self.best_score}