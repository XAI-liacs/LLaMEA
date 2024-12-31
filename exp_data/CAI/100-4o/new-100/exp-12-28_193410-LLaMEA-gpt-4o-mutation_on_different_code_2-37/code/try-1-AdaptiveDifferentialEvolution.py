import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population_size = 10 * self.dim
        F = 0.8  # Mutation factor
        CR = 0.9  # Initial crossover rate
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Select three random indices different from i
                idxs = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                a, b, c = pop[idxs]

                # Mutation with dynamic F
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.rand() < CR or j == self.dim - 1:
                        trial[j] = mutant[j]

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial

                # Update global best
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                # Adapt crossover probability based on success rate
                CR = 0.9 * (1 - (evaluations / self.budget))
                
                # Adjust mutation factor F over time
                F = 0.8 * (1 - (evaluations / self.budget))

        return self.f_opt, self.x_opt