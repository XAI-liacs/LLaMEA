import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F = 0.8  # differential weight
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

                # Mutation
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
                    
                    # Local Search refinement
                    trial_local = self.local_search(trial, func)
                    trial_local_fitness = func(trial_local)
                    num_evals += 1
                    if trial_local_fitness < trial_fitness:
                        population[i] = trial_local
                        fitness[i] = trial_local_fitness

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        # Simple local search using a small perturbation
        best_local = x
        best_local_fitness = func(x)
        perturbation = 0.05 * (func.bounds.ub - func.bounds.lb)

        for _ in range(5):  # 5 local steps
            perturb = np.random.uniform(-perturbation, perturbation, self.dim)
            candidate = np.clip(x + perturb, func.bounds.lb, func.bounds.ub)
            candidate_fitness = func(candidate)
            if candidate_fitness < best_local_fitness:
                best_local = candidate
                best_local_fitness = candidate_fitness

        return best_local