import numpy as np

class MAHO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 2)
        self.mutation_factor = 0.5 + 0.3 * np.random.rand()  # Scaled mutation factor
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
            # Adaptive mutation factor adjustment
            # Change: Introduced a layer-specific mutation factor
            layer_specific_factor = np.random.rand(self.dim)  # New line added
            self.mutation_factor = 0.5 + 0.5 * (self.evaluations / self.budget) * layer_specific_factor
            mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            self.crossover_prob = 0.5 + 0.2 * np.random.rand()  # Adaptive crossover probability scaling
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
        dynamic_local_search_prob = self.local_search_prob + 0.2 * (self.evaluations / self.budget)  # Dynamic local search probability
        if np.random.rand() < dynamic_local_search_prob:
            perturbation = np.random.uniform(-self.robustness_factor, self.robustness_factor, x.shape)
            x_perturbed = np.clip(x + perturbation, func.bounds.lb, func.bounds.ub)
            perturbed_cost = func(x_perturbed)
            self.evaluations += 1
            if perturbed_cost < func(x):
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