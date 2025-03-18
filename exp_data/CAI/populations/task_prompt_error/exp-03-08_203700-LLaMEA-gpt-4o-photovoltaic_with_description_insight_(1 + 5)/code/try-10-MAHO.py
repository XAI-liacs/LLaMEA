import numpy as np

class MAHO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7
        self.local_search_prob = 0.3
        self.layers_increment = 5  # Increment layers gradually
        self.robustness_factor = 0.05  # Small perturbation factor
        self.evaluations = 0

    def differential_evolution(self, pop, func):
        next_pop = []
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = pop[np.random.choice(indices, 3, replace=False)]
            adaptive_mutation_factor = self.mutation_factor + np.random.uniform(-0.1, 0.1)  # Adaptive mutation
            mutant = np.clip(a + adaptive_mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_prob
            trial = np.where(cross_points, mutant, pop[i])
            trial_cost = func(trial)
            self.evaluations += 1
            if trial_cost < func(pop[i]):
                next_pop.append(trial)
            else:
                next_pop.append(pop[i])
        return np.array(next_pop)

    def local_search(self, x, func):
        if np.random.rand() < self.local_search_prob:
            perturbation = np.random.uniform(-self.robustness_factor, self.robustness_factor, x.shape)
            x_perturbed = np.clip(x + perturbation, func.bounds.lb, func.bounds.ub)
            layer_perturbed_cost = func(x_perturbed)
            self.evaluations += 1
            if layer_perturbed_cost < func(x):
                return x_perturbed
        return x

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        while self.evaluations < self.budget:
            pop = self.differential_evolution(pop, func)
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                pop[i] = self.local_search(pop[i], func)
            if self.dim < 32 and self.evaluations / self.budget > 0.5:
                self.dim = min(32, self.dim + self.layers_increment)
        best_idx = np.argmin([func(ind) for ind in pop])
        return pop[best_idx]