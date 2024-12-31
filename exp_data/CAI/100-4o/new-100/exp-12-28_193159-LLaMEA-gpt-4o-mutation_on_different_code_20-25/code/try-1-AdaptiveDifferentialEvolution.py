import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.initial_population_size = max(5, 20)
        self.population_size = self.initial_population_size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        if np.min(fitness) < self.f_opt:
            self.f_opt = np.min(fitness)
            self.x_opt = population[np.argmin(fitness)]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                self.F = 0.5 + np.random.rand() * 0.5  # dynamic mutation rate
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Evaluate trial vector
                f_trial = func(trial)
                evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break

            # Adaptive local search and adaptive population size
            if evaluations < self.budget:
                best_idx = np.argmin(fitness)
                local_step = np.random.normal(0, 0.1, self.dim)
                local_candidate = np.clip(population[best_idx] + local_step, lb, ub)
                f_local_candidate = func(local_candidate)
                evaluations += 1
                
                if f_local_candidate < self.f_opt:
                    self.f_opt = f_local_candidate
                    self.x_opt = local_candidate

            # Adjusting population size for adaptive search exploration
            if evaluations % (self.budget // 10) == 0 and self.population_size < 50:
                self.population_size += 1
                new_individual = np.random.uniform(lb, ub, self.dim)
                population = np.vstack((population, new_individual))
                fitness = np.append(fitness, func(new_individual))
                evaluations += 1

        return self.f_opt, self.x_opt