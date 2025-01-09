import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 20
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        self.w = 0.7   # Inertia weight
        self.F = 0.5   # Differential weight
        self.CR = 0.9  # Crossover probability
        self.pop = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest = np.copy(self.pop)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest = None
        self.gbest_score = np.inf
        self.eval_count = 0

    def evaluate(self, func, individual):
        if self.eval_count < self.budget:
            score = func(individual)
            self.eval_count += 1
            return score
        else:
            return np.inf

    def __call__(self, func):
        while self.eval_count < self.budget:
            # Particle Swarm Optimization component
            for i in range(self.population_size):
                score = self.evaluate(func, self.pop[i])
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest[i] = self.pop[i]
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.pop[i]

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.vel = (self.w * self.vel + 
                        self.c1 * r1 * (self.pbest - self.pop) + 
                        self.c2 * r2 * (self.gbest - self.pop))
            self.pop = np.clip(self.pop + self.vel, self.lb, self.ub)

            # Differential Evolution component
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.pop[i])
                trial_score = self.evaluate(func, trial)
                if trial_score < self.pbest_scores[i]:
                    self.pop[i] = trial
                    self.pbest_scores[i] = trial_score
                    if trial_score < self.gbest_score:
                        self.gbest_score = trial_score
                        self.gbest = trial

        return self.gbest