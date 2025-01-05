import numpy as np
from cma import CMAEvolutionStrategy

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F_min, self.F_max = 0.5, 1.0  # adaptive differential weight range
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        while num_evals < self.budget:
            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Mutation with adaptive F
                self.F = self.adaptive_differential_weight(fitness)
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Local Search refinement using CMA-ES
                    trial_local = self.cma_es_local_search(trial, func)
                    trial_local_fitness = func(trial_local)
                    num_evals += 1
                    if trial_local_fitness < trial_fitness:
                        population[i] = trial_local
                        fitness[i] = trial_local_fitness

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt

    def adaptive_differential_weight(self, fitness):
        fitness_std = np.std(fitness)
        return self.F_min + (self.F_max - self.F_min) * fitness_std / (fitness_std + 1e-9)

    def cma_es_local_search(self, x, func):
        es = CMAEvolutionStrategy(x, 0.1, {'bounds': [func.bounds.lb, func.bounds.ub], 'popsize': 5})
        while not es.stop() and self.budget > 0:
            solutions = es.ask()
            es.tell(solutions, [func(sol) for sol in solutions])
            self.budget -= len(solutions)
        return es.result.xbest