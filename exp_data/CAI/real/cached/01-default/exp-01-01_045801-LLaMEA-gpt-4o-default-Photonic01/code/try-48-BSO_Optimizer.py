import numpy as np

class BSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alph = 0.8
        self.beta = 1.2
        self.mutualism_factor = 0.5
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

    def update_best(self):
        best_idx = np.argmin(self.scores)
        self.best_solution = self.population[best_idx].copy()
        self.best_score = self.scores[best_idx]

    def mutualism(self, ind_a, ind_b):
        mutual_vector = (ind_a + ind_b) / 2
        mutual_a = ind_a + np.random.uniform() * (self.best_solution - mutual_vector) * self.mutualism_factor
        mutual_b = ind_b + np.random.uniform() * (self.best_solution - mutual_vector) * self.mutualism_factor
        return mutual_a, mutual_b

    def commensalism(self, ind_a):
        random_neighbor = self.population[np.random.randint(self.population_size)]
        commensal = ind_a + np.random.uniform(-1, 1) * (self.best_solution - random_neighbor)
        return commensal

    def parasitism(self, ind_a, lb, ub):
        parasite = ind_a.copy()
        idx = np.random.randint(self.dim)
        parasite[idx] = np.random.uniform(lb[idx], ub[idx])
        return parasite

    def __call__(self, func):
        self.func = func
        lb, ub = func.bounds.lb, func.bounds.ub

        self.initialize_population(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                partner_idx = np.random.randint(self.population_size)
                if i != partner_idx:
                    ind_a = self.population[i]
                    ind_b = self.population[partner_idx]

                    mutual_a, mutual_b = self.mutualism(ind_a, ind_b)
                    commensal_a = self.commensalism(ind_a)
                    parasite_b = self.parasitism(ind_b, lb, ub)

                    candidates = [mutual_a, mutual_b, commensal_a, parasite_b]
                    candidates_scores = [self.evaluate(np.clip(c, lb, ub)) for c in candidates]

                    if min(candidates_scores) < self.scores[i]:
                        best_candidate_idx = np.argmin(candidates_scores)
                        self.population[i] = np.clip(candidates[best_candidate_idx], lb, ub)
                        self.scores[i] = candidates_scores[best_candidate_idx]

                    self.evaluations += len(candidates)
                    self.update_best()

        return {'solution': self.best_solution, 'fitness': self.best_score}