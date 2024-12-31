import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation: select 3 random indices and create a mutant vector
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, lb, ub)

                # Crossover: create trial vector
                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                # Evaluate trial vector
                trial_fitness = func(trial)
                eval_count += 1

                # Selection: replace if trial is better
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best if necessary
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

        return self.f_opt, self.x_opt