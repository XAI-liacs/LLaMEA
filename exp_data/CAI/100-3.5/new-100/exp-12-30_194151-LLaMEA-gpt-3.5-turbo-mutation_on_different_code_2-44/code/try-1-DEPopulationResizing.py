class DEPopulationResizing:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.9, f_adapt=0.1, cr_adapt=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.f_adapt = f_adapt
        self.cr_adapt = cr_adapt
        self.f_opt = np.Inf
        self.x_opt = None

    def evolve_population(self, func, population):
        new_population = []
        for i, target in enumerate(population):
            candidates = [ind for ind in population if ind is not target]
            a, b, c = np.random.choice(candidates, 3, replace=False)
            f_curr = self.f + np.random.normal(0, self.f_adapt)
            cr_curr = np.clip(self.cr + np.random.normal(0, self.cr_adapt), 0, 1)
            mutant = np.clip(a + f_curr * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < cr_curr
            trial = np.where(cross_points, mutant, target)
            if func(trial) < func(target):
                new_population.append(trial)
            else:
                new_population.append(target)
        return new_population