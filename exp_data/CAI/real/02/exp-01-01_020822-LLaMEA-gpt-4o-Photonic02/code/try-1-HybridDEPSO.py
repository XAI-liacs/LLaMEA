import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.c1 = 1.5  # PSO cognitive coefficient
        self.c2 = 1.5  # PSO social coefficient
        self.population = None
        self.velocity = None

    def initialize_population(self, bounds):
        self.population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
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

    def de_mutation_and_crossover(self, idx, bounds):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        crossover = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover, mutant, self.population[idx])
        return trial

    def pso_update_velocity(self, idx):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = self.c1 * r1 * (self.personal_best[idx] - self.population[idx])
        social = self.c2 * r2 * (self.global_best - self.population[idx])
        self.velocity[idx] = 0.5 * self.velocity[idx] + cognitive + social

    def local_search(self, individual, bounds):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        new_individual = np.clip(individual + perturbation, bounds.lb, bounds.ub)
        return new_individual

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        eval_count = 0
        
        while eval_count < self.budget:
            scores = self.evaluate_population(func)
            eval_count += self.population_size
            
            for i in range(self.population_size):
                trial = self.de_mutation_and_crossover(i, bounds)
                trial_score = func(trial)
                eval_count += 1
                if trial_score < scores[i]:
                    self.population[i] = trial
                    scores[i] = trial_score

                if eval_count >= self.budget:
                    break

            # Local search exploit phase
            if eval_count < self.budget:
                for i in range(self.population_size):
                    local_trial = self.local_search(self.population[i], bounds)
                    local_score = func(local_trial)
                    eval_count += 1
                    if local_score < scores[i]:
                        self.population[i] = local_trial
                        scores[i] = local_score
                    
                    if eval_count >= self.budget:
                        break

            # PSO update
            for i in range(self.population_size):
                self.pso_update_velocity(i)
                self.population[i] += self.velocity[i]
                self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)

        return self.global_best, self.global_best_score