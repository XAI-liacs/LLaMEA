import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)  # Maintain adaptive population size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Initial scaling factor
        self.CR = 0.9  # Initial crossover rate

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.population_size
        stagnation_count = 0  # Track if population is stuck

        while evals < self.budget:
            if stagnation_count > self.population_size:
                # Restart mechanism for better diversity
                pop = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
                fitness = np.array([func(ind) for ind in pop])
                evals += self.population_size
                stagnation_count = 0
                continue

            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation with self-adaptive F
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                self.F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
                mutant = np.clip(a + self.F * (b - c), bounds[0], bounds[1])

                # Crossover with self-adaptive CR
                trial = np.copy(pop[i])
                self.CR = np.clip(np.random.normal(self.CR, 0.1), 0.5, 1.0)
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection with stagnation tracking
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    stagnation_count = 0  # Reset stagnation count
                else:
                    stagnation_count += 1  # Increment if no improvement
                
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

        return self.f_opt, self.x_opt