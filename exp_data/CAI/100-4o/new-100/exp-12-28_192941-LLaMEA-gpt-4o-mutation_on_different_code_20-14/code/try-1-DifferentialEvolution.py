import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.8, Cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.Cr = Cr  # Crossover probability
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        # Update the best solution found so far
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Main loop
        while evals < self.budget:
            new_population = np.copy(population)  # To implement elitism
            for i in range(self.pop_size):
                # Mutation: select three indices a, b, c different from i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                # Adaptive mutation factor
                adaptive_F = self.F + (np.random.rand() * 0.2 - 0.1)

                # Calculate mutant vector
                mutant = population[a] + adaptive_F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        trial[j] = mutant[j]

                # Selection
                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    new_population[i] = trial  # Use new_population for elitism
                    fitness[i] = f_trial

                    # Update the best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Break if budget is exceeded
                if evals >= self.budget:
                    break

            population = new_population  # Apply elitism by carrying over the best solutions

        return self.f_opt, self.x_opt