import numpy as np

class DEAM:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.pop_size

        while eval_count < self.budget:
            # Calculate diversity
            diversity = np.std(pop, axis=0).mean()
            mutation_factor = max(0.5, min(1.0, diversity))

            # DE Mutation and Crossover
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = np.clip(x1 + mutation_factor * (x2 - x3), func.bounds.lb, func.bounds.ub)
                crossover_rate = 0.9
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial vector
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt