import numpy as np

class PSODE:
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=1.5, c2=1.5, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.F = F
        self.CR = CR
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.apply_along_axis(func, 1, pbest_pos)
        gbest_pos = pbest_pos[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)

        evals = self.pop_size
        vel_max = (self.upper_bound - self.lower_bound) / 2

        while evals < self.budget:
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.c1 *= 0.98
            inertia_weight = 0.9 - 0.5 * (evals / self.budget)  # Dynamic inertia weight
            vel = inertia_weight * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -vel_max, vel_max)
            pos = pos + vel
            pos = np.clip(pos, self.lower_bound, self.upper_bound)

            fitness = np.apply_along_axis(func, 1, pos)
            evals += self.pop_size

            better_idx = fitness < pbest_val
            pbest_pos[better_idx] = pos[better_idx]
            pbest_val[better_idx] = fitness[better_idx]

            if np.min(fitness) < gbest_val:
                gbest_val = np.min(fitness)
                gbest_pos = pos[np.argmin(fitness)]

            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                idxs = np.arange(self.pop_size)
                idxs = idxs[idxs != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                adaptive_F = self.F * (1 - evals / self.budget)
                mutant = np.clip(pbest_pos[a] + adaptive_F * (pbest_pos[b] - pbest_pos[c]), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pos[i])
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    pos[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < pbest_val[i]:
                        pbest_pos[i] = trial
                        pbest_val[i] = trial_fitness
                        if trial_fitness < gbest_val:
                            gbest_val = trial_fitness
                            gbest_pos = trial

            vel *= 0.95 + 0.05 * np.std(fitness) / np.max(fitness)
            if np.std(fitness) < 1e-4:
                vel[np.random.rand(self.pop_size, self.dim) < 0.15] = np.random.uniform(-1, 1)

            if np.random.rand() < 0.05:
                gbest_pos += np.random.normal(0, 0.1 * (1 - evals / self.budget), self.dim)
                gbest_pos = np.clip(gbest_pos, self.lower_bound, self.upper_bound)
                self.F = np.clip(self.F * 0.99, 0.4, 0.8)

        return gbest_pos, gbest_val