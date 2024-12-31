import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.update_best_solution(pop, fitness)

        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.f_opt:
                        self.update_best_solution([trial], [trial_fitness])

                # Adaptive parameter control
                self.F = 0.8 + 0.4 * np.abs((fitness[i] - self.f_opt) / self.f_opt) 
                self.CR = 0.9 - 0.5 * (evaluations / self.budget)

        return self.f_opt, self.x_opt

    def update_best_solution(self, pop, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.f_opt:
            self.f_opt = fitness[min_idx]
            self.x_opt = pop[min_idx]