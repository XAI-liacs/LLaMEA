class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != j]
                a, b, c = np.random.choice(indices, 3, replace=False)

                F_val = np.random.uniform(0.2, 0.8)  # Adaptive F parameter
                CR_val = np.random.uniform(0.7, 1.0)  # Adaptive CR parameter

                mutant = population[a] + F_val * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR_val
                trial_vector = np.where(crossover, mutant, population[j])

                f_trial = func(trial_vector)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial_vector

        return self.f_opt, self.x_opt