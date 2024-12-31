import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Mutation: select three random distinct indices different from i
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive mutation strategy
                F_dynamic = self.F + np.random.rand() * 0.2  # Adjust mutation strength
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                f_trial = func(trial)
                eval_count += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                if eval_count >= self.budget:
                    break

            if eval_count < self.budget // 2:  # Dynamically adjust population size
                self.population_size = max(10, self.population_size - 1)

        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]