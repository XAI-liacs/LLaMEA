import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5
        self.CR = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.population = None
        self.velocity = None

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best = self.population.copy()
        self.global_best = self.population[0].copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_score = np.inf

    def evaluate_population(self, func):
        scores = np.array([func(ind) for ind in self.population])
        for i in range(self.population_size):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best[i] = self.population[i].copy()
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best = self.population[i].copy()
        return scores

    def de_mutation_and_crossover(self, idx, bounds, gen):
        F = self.F_base + 0.25 * np.sin(np.pi * gen / (2 * self.budget))  # Enhanced adaptation
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
        self.CR = 0.85 - 0.15 * np.cos(np.pi * gen / self.budget)  # Adjusted adaptive crossover rate
        crossover = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover, mutant, self.population[idx])
        return trial

    def pso_update_velocity(self, idx):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = (self.c1 + 0.2 * np.sin(np.pi * idx / self.population_size)) * r1 * (self.personal_best[idx] - self.population[idx])  # More adaptation
        social = self.c2 * r2 * (self.global_best - self.population[idx])
        inertia_weight = 0.4 + 0.6 * np.cos(np.pi * idx / self.population_size)  # Adjusted inertia weight
        self.velocity[idx] = inertia_weight * self.velocity[idx] + cognitive + social

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        gen = 0
        
        while eval_count < self.budget:
            scores = self.evaluate_population(func)
            eval_count += self.population_size
            gen += 1
            
            for i in range(self.population_size):
                trial = self.de_mutation_and_crossover(i, bounds, gen)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score
                    
                if eval_count >= self.budget:
                    break

            for i in range(self.population_size):
                self.pso_update_velocity(i)
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)

        return self.global_best, self.global_best_score