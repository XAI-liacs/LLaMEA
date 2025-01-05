import numpy as np

class AQFPA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.population = None
        self.scores = None
        self.best_solution = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.p = 0.8  # switch probability between global and local pollination
        self.beta_min = 0.5
        self.beta_max = 1.5

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.scores = np.array([self.evaluate(ind) for ind in self.population])
        self.update_best()

    def evaluate(self, solution):
        return self.func(solution)

    def levy_flight(self, beta):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / beta)
        return step

    def global_pollination(self, flower):
        beta = np.random.uniform(self.beta_min, self.beta_max)
        step = self.levy_flight(beta)
        return flower + step * (self.best_solution - flower)

    def local_pollination(self, flower, partner):
        epsilon = np.random.uniform(0, 1, self.dim)
        return flower + epsilon * (partner - flower)

    def select(self, target_idx, trial):
        trial_score = self.evaluate(trial)
        if trial_score < self.scores[target_idx]:
            self.population[target_idx] = trial
            self.scores[target_idx] = trial_score
            if trial_score < self.best_score:
                self.best_score = trial_score
                self.best_solution = trial.copy()

    def quantum_superposition(self, flower):
        quantum_superposition = np.random.uniform(-1, 1, self.dim)
        quantum_flower = self.best_solution + quantum_superposition * np.abs(flower - self.best_solution)
        return np.clip(quantum_flower, self.func.bounds.lb, self.func.bounds.ub)

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
                flower = self.population[i]
                if np.random.rand() < self.p:
                    trial = self.global_pollination(flower)
                else:
                    partner_idx = np.random.randint(0, self.population_size)
                    while partner_idx == i:
                        partner_idx = np.random.randint(0, self.population_size)
                    trial = self.local_pollination(flower, self.population[partner_idx])

                trial = np.clip(trial, lb, ub)
                self.select(i, trial)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

            for i in range(self.population_size):
                quantum_flower = self.quantum_superposition(self.population[i])
                self.select(i, quantum_flower)
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break

        return {'solution': self.best_solution, 'fitness': self.best_score}