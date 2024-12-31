import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.pop_size

        while evaluations < self.budget:
            if evaluations % (self.budget // 10) == 0:  # Resize population dynamically
                self.pop_size = max(4, self.pop_size // 2)
                pop = pop[:self.pop_size]
                fitness = fitness[:self.pop_size]

            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                F = np.random.uniform(0.5, 1.0)  # Adaptive differential weight
                CR = np.random.uniform(0.1, 0.9)  # Adaptive crossover rate
                mutant = np.clip(a + F * (b - c), lb, ub)
                trial = np.copy(pop[i])
                
                j_rand = np.random.randint(0, self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt