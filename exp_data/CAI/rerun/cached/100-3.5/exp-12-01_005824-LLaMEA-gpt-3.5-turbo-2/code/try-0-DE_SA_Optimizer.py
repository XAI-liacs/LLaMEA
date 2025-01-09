import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def mutated_vector(pop, f):
            idx = np.random.choice(len(pop), 3, replace=False)
            v = pop[idx[0]] + f * (pop[idx[1]] - pop[idx[2]])
            return np.clip(v, self.lb, self.ub)

        pop_size = 10 * self.dim
        pop = np.random.uniform(self.lb, self.ub, (pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        
        f = 0.5
        cr = 0.9
        T = 1.0
        alpha = 0.95

        for _ in range(self.budget):
            new_pop = np.zeros_like(pop)
            for i in range(pop_size):
                trial = pop[i]
                idxs = np.random.choice(pop_size, 3, replace=False)
                for j in range(self.dim):
                    if np.random.rand() < cr or j == np.random.randint(0, self.dim):
                        trial[j] = pop[idxs[0], j] + f * (pop[idxs[1], j] - pop[idxs[2], j])
                trial = np.clip(trial, self.lb, self.ub)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < fitness[best_idx]:
                        best_idx = i
                        best = trial
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

            T *= alpha
            T = max(T, 1e-10)
            for i in range(self.dim):
                diff = np.random.uniform(-0.5, 0.5)
                best[i] += diff
            best = np.clip(best, self.lb, self.ub)

        return best