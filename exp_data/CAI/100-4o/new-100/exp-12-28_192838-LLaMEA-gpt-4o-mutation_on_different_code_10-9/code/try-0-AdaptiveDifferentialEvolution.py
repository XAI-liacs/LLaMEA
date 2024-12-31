import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.local_search_prob = 0.1

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation and Crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation
                f = func(trial)
                eval_count += 1

                # Selection
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                # Local search with probability
                if np.random.rand() < self.local_search_prob:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(trial + perturbation, lb, ub)
                    f_local = func(local_trial)
                    eval_count += 1

                    if f_local < fitness[i]:
                        fitness[i] = f_local
                        population[i] = local_trial
                        if f_local < self.f_opt:
                            self.f_opt = f_local
                            self.x_opt = local_trial

        return self.f_opt, self.x_opt