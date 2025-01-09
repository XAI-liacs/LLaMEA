import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(10 * dim, budget // 2)
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def local_search(self, x, func, step_size=0.1):
        best_x = np.copy(x)
        best_f = func(best_x)
        for d in range(self.dim):
            x_new = np.copy(best_x)
            x_new[d] += step_size
            if x_new[d] > self.upper_bound:
                x_new[d] = self.upper_bound
            elif x_new[d] < self.lower_bound:
                x_new[d] = self.lower_bound
            f_new = func(x_new)
            if f_new < best_f:
                best_x, best_f = x_new, f_new
        return best_x, best_f

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        eval_count = self.population_size
        while eval_count < self.budget:
            # Differential Evolution
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                eval_count += 1

                # Selection
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

            # Local Search
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                improved_x, improved_f = self.local_search(pop[i], func)
                eval_count += self.dim  # Account for local search evaluations

                if improved_f < fitness[i]:
                    pop[i] = improved_x
                    fitness[i] = improved_f

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]