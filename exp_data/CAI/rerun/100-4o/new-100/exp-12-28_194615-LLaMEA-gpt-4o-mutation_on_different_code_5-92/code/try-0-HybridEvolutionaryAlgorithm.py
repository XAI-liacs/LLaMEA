import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget=10000, dim=10, pop_size=50, cr=0.9, f=0.8):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.cr = cr
        self.f = f
        self.f_opt = np.Inf
        self.x_opt = None
        self.eval_count = 0

    def local_search(self, x, func):
        best_x = x
        best_f = func(x)
        step_size = 0.01
        for _ in range(10):  # limited local search steps
            for i in range(self.dim):
                trial_x = np.copy(best_x)
                trial_x[i] += step_size * np.random.randn()
                trial_f = func(trial_x)
                self.eval_count += 1
                if trial_f < best_f:
                    best_f = trial_f
                    best_x = trial_x
                if self.eval_count >= self.budget:
                    break
            if self.eval_count >= self.budget:
                break
        return best_f, best_x

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                
                # Differential Evolution
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial individual
                trial_f = func(trial)
                self.eval_count += 1

                # Selection
                if trial_f < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_f

            # Update global best
            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]

            # Local Search on best individual so far
            if self.eval_count < self.budget:
                self.f_opt, self.x_opt = self.local_search(self.x_opt, func)

        return self.f_opt, self.x_opt