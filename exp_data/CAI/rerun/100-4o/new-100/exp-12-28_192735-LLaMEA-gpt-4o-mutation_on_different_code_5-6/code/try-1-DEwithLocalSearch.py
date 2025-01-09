import numpy as np

class DEwithLocalSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.bounds = (-5.0, 5.0)

    def mutate(self, pop, idx):
        r1, r2, r3 = np.random.choice([i for i in range(self.population_size) if i != idx], 3, replace=False)
        mutant = pop[r1] + self.mutation_factor * (pop[r2] - pop[r3])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, candidate, func):
        step_size = 0.1 * (self.bounds[1] - self.bounds[0]) * (self.f_opt - func(candidate))
        for _ in range(5):
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_candidate = np.clip(candidate + perturbation, self.bounds[0], self.bounds[1])
            f_new = func(new_candidate)
            if f_new < func(candidate):
                candidate = new_candidate
                if f_new < self.f_opt:
                    self.f_opt = f_new
                    self.x_opt = candidate
        return candidate

    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                mutant = self.mutate(pop, i)
                trial = self.crossover(pop[i], mutant)
                f_trial = func(trial)

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                pop[i] = self.local_search(pop[i], func)
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = pop[i]
        
        return self.f_opt, self.x_opt