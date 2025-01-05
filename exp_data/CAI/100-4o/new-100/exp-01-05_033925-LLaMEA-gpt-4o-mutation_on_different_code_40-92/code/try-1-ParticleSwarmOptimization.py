import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, n_particles=30):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.f_opt = np.Inf
        self.x_opt = None
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_base = 2.5  # Adjust base cognitive factor
        self.c2_base = 1.5  # Adjust base social factor
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.n_particles, self.dim))
        v = np.zeros((self.n_particles, self.dim))
        personal_best = np.copy(x)
        personal_best_values = np.array([func(ind) for ind in x])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        iter_count = self.budget // self.n_particles
        for t in range(iter_count):
            c1 = self.c1_base * (0.5 + np.sin(3 * np.pi * t / iter_count))
            c2 = self.c2_base * (0.5 + np.cos(3 * np.pi * t / iter_count))
            w = self.w_max - (self.w_max - self.w_min) * (t / iter_count)
            for i in range(self.n_particles):
                r1 = np.random.uniform(size=self.dim)
                r2 = np.random.uniform(size=self.dim)
                
                v[i] = (w * v[i] +
                        c1 * r1 * (personal_best[i] - x[i]) +
                        c2 * r2 * (global_best - x[i]))

                x[i] = np.clip(x[i] + v[i], lb, ub)

                f_value = func(x[i])
                if f_value < personal_best_values[i]:
                    personal_best[i] = x[i]
                    personal_best_values[i] = f_value
                
                if f_value < global_best_value:
                    global_best = x[i]
                    global_best_value = f_value

            # Chaotic local search
            if t % 10 == 0:
                for i in range(self.n_particles):
                    chaos_factor = 0.1 * (np.random.rand(self.dim) - 0.5)
                    new_solution = np.clip(global_best + chaos_factor, lb, ub)
                    new_value = func(new_solution)
                    if new_value < global_best_value:
                        global_best = new_solution
                        global_best_value = new_value

        self.f_opt = global_best_value
        self.x_opt = global_best
        return self.f_opt, self.x_opt