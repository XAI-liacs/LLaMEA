import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_spent = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        while budget_spent < self.budget:
            for i in range(self.pop_size):
                # Select three distinct indices
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation: Generate a donor vector
                donor = population[a] + self.F * (population[b] - population[c])
                donor = np.clip(donor, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                cross_points = np.random.rand(self.dim) < self.CR
                trial[cross_points] = donor[cross_points]

                # Selection: Evaluate trial vector
                f_trial = func(trial)
                budget_spent += 1

                # If trial is better, replace old solution
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                    # Update global best
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

                # Break if budget is exhausted
                if budget_spent >= self.budget:
                    break

        return self.f_opt, self.x_opt