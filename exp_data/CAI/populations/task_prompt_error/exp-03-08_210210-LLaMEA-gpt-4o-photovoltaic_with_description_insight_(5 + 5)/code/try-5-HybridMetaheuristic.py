import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * self.dim
        self.f = 0.8  # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.local_search_steps = 5
        self.robustness_factor = 0.01

    def differential_evolution(self, func, pop, lb, ub):
        new_pop = np.copy(pop)
        for i in range(self.pop_size):
            if self.eval_count >= self.budget:
                break
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            fitness_trial = func(trial) - self.robustness_factor * np.std(trial)
            self.eval_count += 1
            if fitness_trial < func(pop[i]):
                new_pop[i] = trial
        return new_pop

    def local_search(self, func, individual, lb, ub):
        best = np.copy(individual)
        best_fitness = func(best)
        for _ in range(self.local_search_steps):
            if self.eval_count >= self.budget:
                break
            candidate = best + np.random.normal(0, 0.1, size=self.dim)
            candidate = np.clip(candidate, lb, ub)
            fitness_candidate = func(candidate) - self.robustness_factor * np.std(candidate)
            self.eval_count += 1
            if fitness_candidate < best_fitness:
                best, best_fitness = candidate, fitness_candidate
        return best

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        self.eval_count = 0

        while self.eval_count < self.budget:
            pop = self.differential_evolution(func, pop, lb, ub)
            for i in range(self.pop_size):
                if self.eval_count >= self.budget:
                    break
                pop[i] = self.local_search(func, pop[i], lb, ub)
                layer_expansion = min(self.dim, int(self.eval_count / self.budget * self.dim))
                expanded_lb = lb[:layer_expansion] if layer_expansion > 0 else lb
                expanded_ub = ub[:layer_expansion] if layer_expansion > 0 else ub
                pop[i][:layer_expansion] = self.local_search(func, pop[i][:layer_expansion], expanded_lb, expanded_ub)

        best_idx = np.argmin([func(individual) for individual in pop])
        return pop[best_idx]