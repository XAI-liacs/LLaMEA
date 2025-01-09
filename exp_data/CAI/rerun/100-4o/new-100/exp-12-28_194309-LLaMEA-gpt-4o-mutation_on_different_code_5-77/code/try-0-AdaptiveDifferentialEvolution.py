import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population within bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.budget -= self.pop_size

        while self.budget > 0:
            for i in range(self.pop_size):
                # Mutation: Create mutant vector
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover: Create trial vector
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])

                # Selection: Choose between trial and parent
                trial_fitness = func(trial)
                self.budget -= 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                # Update best solution found
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

                # Break if budget is exhausted
                if self.budget <= 0:
                    break

            # Adaptive parameter control
            self.F = np.random.normal(loc=0.5, scale=0.3)
            self.CR = np.random.uniform(0.5, 1.0)

        return self.f_opt, self.x_opt