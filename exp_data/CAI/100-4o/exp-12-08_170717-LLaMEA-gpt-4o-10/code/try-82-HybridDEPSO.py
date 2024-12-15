import numpy as np

class HybridDEPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 50
        self.cr = 0.9
        self.f = 0.8
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = pop.copy()
        personal_best_scores = np.array([func(x) for x in pop])
        
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()

        evals = self.pop_size
        stagnation_counter = 0

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.f * (b - c)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < personal_best_scores[i]:
                    personal_best_scores[i] = trial_fitness
                    personal_best_positions[i] = trial

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            if evals >= self.budget:
                break
            self.w = 0.9 - (evals / self.budget) * 0.4
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            pop = np.clip(pop + velocities, lb, ub)

            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                fitness = func(pop[i])
                evals += 1

                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = pop[i]

                    if fitness < self.f_opt:
                        self.f_opt = fitness
                        self.x_opt = pop[i]
                        global_best_position = pop[i]
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

            if stagnation_counter > self.pop_size * 2: 
                pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
                stagnation_counter = 0

        return self.f_opt, self.x_opt