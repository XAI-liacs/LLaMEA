import numpy as np

class DE_LocalSearch:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_evals = 0
        self.stagnation_counter = 0

    def mutate(self, idx, population):
        idxs = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, -5.0, 5.0)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def local_search(self, x, func):
        step_size = 0.1
        best_x = x
        best_f = func(x)
        self.num_evals += 1
        improved = False
        for _ in range(10):
            neighbor = x + np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(neighbor, -5.0, 5.0)
            f = func(neighbor)
            self.num_evals += 1
            if f < best_f:
                best_f = f
                best_x = neighbor
                improved = True
            if self.num_evals >= self.budget:
                break
        if not improved:
            step_size *= 0.5  # Adaptive step-size control
        return best_x, best_f

    def reinitialize_population(self, population):
        for i in range(self.pop_size):
            if np.random.rand() < 0.1:
                population[i] = np.random.uniform(-5.0, 5.0, self.dim)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.num_evals += self.pop_size

        while self.num_evals < self.budget:
            for i in range(self.pop_size):
                if self.num_evals >= self.budget:
                    break

                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                self.num_evals += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    self.stagnation_counter = 0
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    self.stagnation_counter += 1

                if self.num_evals + 10 < self.budget:
                    local_x, local_f = self.local_search(population[i], func)
                    if local_f < fitness[i]:
                        population[i] = local_x
                        fitness[i] = local_f
                        if local_f < self.f_opt:
                            self.f_opt = local_f
                            self.x_opt = local_x
            
            if self.stagnation_counter > self.pop_size * 2:
                self.reinitialize_population(population)
                self.stagnation_counter = 0

        return self.f_opt, self.x_opt