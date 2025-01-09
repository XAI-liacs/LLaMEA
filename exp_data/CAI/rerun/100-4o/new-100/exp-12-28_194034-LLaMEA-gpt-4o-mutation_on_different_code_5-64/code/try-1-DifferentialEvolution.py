import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                # Select three distinct random indices
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Mutation: generate a mutant vector
                F = np.random.uniform(0.6, 1.0)  # Mutation factor
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)

                # Crossover: create a trial vector
                CR = np.random.uniform(0.2, 0.9)  # Crossover probability
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate the trial vector
                trial_fitness = func(trial)
                evals += 1

                # Selection: replace if trial is better
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt