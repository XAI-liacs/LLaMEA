import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.w = 0.7298  # inertia coefficient
        self.c1 = 1.49618  # personal best coefficient
        self.c2 = 1.49618  # global best coefficient
        self.cr = 0.5  # crossover probability
        self.f = 0.5  # differential weight
        self.w_min = 0.4  # minimum inertia coefficient
        self.w_max = 0.9  # maximum inertia coefficient

    def __call__(self, func):
        # Initialize population
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        v = np.zeros((self.population_size, self.dim))
        pbest = x.copy()
        f_values = np.array([func(xi) for xi in x])
        gbest = pbest[np.argmin(f_values)]
        gbest_value = np.min(f_values)

        # Main loop
        for i in range(self.budget - self.population_size):
            self.w = self.w_min + (self.w_max - self.w_min) * (1 - i / (self.budget - self.population_size))  # adaptive inertia coefficient
            for j in range(self.population_size):
                # Particle swarm optimization
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                v[j] = self.w * v[j] + self.c1 * r1 * (pbest[j] - x[j]) + self.c2 * r2 * (gbest - x[j])
                x_new = x[j] + v[j]

                # Differential evolution
                k = np.random.randint(0, self.population_size)
                while k == j:
                    k = np.random.randint(0, self.population_size)
                l = np.random.randint(0, self.population_size)
                while l == j or l == k:
                    l = np.random.randint(0, self.population_size)
                u = x[j] + self.f * (x[k] - x[l]) + 0.1 * np.random.uniform(-1, 1, self.dim)  # improved differential evolution strategy
                u = np.where(np.random.uniform(0, 1, self.dim) < self.cr, u, x[j])

                # Apply bounds
                x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                u = np.clip(u, func.bounds.lb, func.bounds.ub)

                # Evaluate new solutions
                f_new_ps = func(x_new)
                f_new_de = func(u)

                # Update personal best and global best
                if f_new_ps < f_values[j]:
                    pbest[j] = x_new
                    f_values[j] = f_new_ps
                if f_new_de < f_values[j]:
                    pbest[j] = u
                    f_values[j] = f_new_de
                if f_new_ps < gbest_value:
                    gbest = x_new
                    gbest_value = f_new_ps
                if f_new_de < gbest_value:
                    gbest = u
                    gbest_value = f_new_de

                # Update current position
                x[j] = np.where(np.random.uniform(0, 1, self.dim) < 0.5, x_new, u)

            # Update best found solution
            if gbest_value < self.f_opt:
                self.f_opt = gbest_value
                self.x_opt = gbest

        return self.f_opt, self.x_opt