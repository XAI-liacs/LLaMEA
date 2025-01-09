import numpy as np

class DE_CMA_Hybrid:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        mutation_factor = 0.8
        for generation in range(self.budget - self.pop_size):
            # Differential Evolution step
            for j in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Dynamic mutation factor
                mutation_factor = 0.5 + 0.3 * np.sin(0.1 * generation)
                mutant = np.clip(a + mutation_factor * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < 0.9
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[j])

                f_trial = func(trial)
                if f_trial < fitness[j]:
                    fitness[j] = f_trial
                    population[j] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # CMA-ES step: adaptive covariance matrix
            mean = np.mean(population, axis=0)
            cov_matrix = np.cov(population, rowvar=False) + 1e-5 * np.eye(self.dim) * (1 + 0.1 * np.random.rand())
            cma_population = np.random.multivariate_normal(mean, cov_matrix, self.pop_size)

            for k in range(self.pop_size):
                candidate = np.clip(cma_population[k], bounds[0], bounds[1])
                f_candidate = func(candidate)
                if f_candidate < fitness[k]:
                    fitness[k] = f_candidate
                    population[k] = candidate
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = candidate

        return self.f_opt, self.x_opt