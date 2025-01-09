import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * self.dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.T0 = 100  # Initial temperature for simulated annealing
        self.alpha = 0.9  # Cooling rate

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(*self.bounds, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size

        # Main optimization loop
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Mutation using DE/rand/1
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), *self.bounds)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])

                # Adaptation using Simulated Annealing
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                else:
                    # Simulated annealing acceptance criterion
                    T = self.T0 * (self.alpha ** (eval_count // self.population_size))
                    prob = np.exp(-(trial_fitness - fitness[i]) / T)
                    if np.random.rand() < prob:
                        pop[i] = trial
                        fitness[i] = trial_fitness

        return pop[np.argmin(fitness)]