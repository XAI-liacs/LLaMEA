import numpy as np

class AdaptiveDE_CMA:
    def __init__(self, budget, dim, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.population = np.random.uniform(size=(pop_size, dim))
        self.scores = np.full(pop_size, np.inf)
        self.best_position = None
        self.best_score = np.inf
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # DE crossover probability
        self.sigma = 0.3  # CMA-ES initial step size

    def cma_es_step(self, x, grad):
        return x + self.sigma * grad

    def optimize(self, func):
        num_evaluations = 0

        while num_evaluations < self.budget:
            trial_population = np.empty_like(self.population)

            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = self.population[indices]

                # Differential evolution mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])
                trial_population[i] = trial

                trial_score = func(trial)
                num_evaluations += 1

                # Select the better solution
                if trial_score < self.scores[i]:
                    self.scores[i] = trial_score
                    self.population[i] = trial

                # Update global best
                if trial_score < self.best_score:
                    self.best_score = trial_score
                    self.best_position = trial.copy()

            # Apply a CMA-ES step if beneficial
            if num_evaluations / self.budget > 0.5:
                gradients = (trial_population - self.population) * (self.scores[:, None] - self.best_score)
                adaptive_step = np.mean(gradients, axis=0)
                self.population += self.cma_es_step(self.population, adaptive_step)
                self.population = np.clip(self.population, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.optimize(func)
        return self.best_position, self.best_score