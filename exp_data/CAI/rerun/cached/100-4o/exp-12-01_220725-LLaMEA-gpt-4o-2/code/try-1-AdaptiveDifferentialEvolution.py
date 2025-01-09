import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.bounds = (-5.0, 5.0)
        self.eval_count = 0
        
    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.bounds[0], self.bounds[1])

        def adaptive_mutation(pop, best_idx):
            # Adaptive scaling factor based on population diversity
            diversity = np.mean(np.std(pop, axis=0))
            F_adaptive = self.F * (1 + diversity / self.dim)
            F_adaptive = min(max(F_adaptive, 0.1), 0.9)
            idxs = np.arange(self.pop_size)
            trial_vectors = np.zeros_like(pop)
            for i in range(self.pop_size):
                a, b, c = np.random.choice(idxs[idxs != i], 3, replace=False)
                mutant = pop[a] + F_adaptive * (pop[b] - pop[c])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vectors[i] = np.where(cross_points, mutant, pop[i])
                trial_vectors[i] = clip(trial_vectors[i])
            return trial_vectors

        # Initialize population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.eval_count += self.pop_size

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            trial_pop = adaptive_mutation(pop, best_idx)
            trial_fitness = np.array([func(ind) for ind in trial_pop])
            self.eval_count += self.pop_size

            # Selection
            for i in range(self.pop_size):
                if trial_fitness[i] < fitness[i]:
                    fitness[i] = trial_fitness[i]
                    pop[i] = trial_pop[i]

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]