import numpy as np

class HPSO_ADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.w = 0.7  # changed inertia weight
        self.c1 = 1.49618  # personal best weight
        self.c2 = 1.49618  # global best weight
        self.F = 0.8  # changed scaling factor for differential evolution
        self.CR = 0.9  # crossover probability for differential evolution

    def __call__(self, func):
        # Initialize population
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        v = np.zeros((self.population_size, self.dim))
        pbest = x.copy()
        f = np.array([func(xi) for xi in x])
        gbest = x[np.argmin(f)]
        gbest_f = np.min(f)

        # Main loop
        for i in range(self.budget - self.population_size):
            # Particle swarm optimization
            for j in range(self.population_size):
                v[j] = self.w * v[j] + self.c1 * np.random.uniform(0, 1, self.dim) * (pbest[j] - x[j]) + self.c2 * np.random.uniform(0, 1, self.dim) * (gbest - x[j])
                x_new = x[j] + v[j]
                x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)

                # Evaluate new solution
                f_new = func(x_new)
                if f_new < f[j]:
                    x[j] = x_new
                    f[j] = f_new
                    pbest[j] = x_new
                    if f_new < gbest_f:
                        gbest = x_new
                        gbest_f = f_new

            # Differential evolution
            for j in range(self.population_size):
                r1, r2, r3 = np.random.choice([k for k in range(self.population_size) if k!= j], 3, replace=False)
                x_new = x[j] + self.F * (x[r1] - x[r2]) + np.random.uniform(0, 1, self.dim) * (pbest[r3] - x[j])
                x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                u = np.copy(x_new)
                for k in range(self.dim):
                    if np.random.uniform(0, 1) < self.CR:
                        u[k] = x_new[k]
                    else:
                        u[k] = x[j, k]

                # Evaluate new solution
                f_new = func(u)
                if f_new < f[j]:
                    x[j] = u
                    f[j] = f_new
                    pbest[j] = u
                    if f_new < gbest_f:
                        gbest = u
                        gbest_f = f_new

        self.f_opt = gbest_f
        self.x_opt = gbest
        return self.f_opt, self.x_opt