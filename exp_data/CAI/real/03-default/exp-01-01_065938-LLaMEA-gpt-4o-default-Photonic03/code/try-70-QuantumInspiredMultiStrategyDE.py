import numpy as np

class QuantumInspiredMultiStrategyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.F_min, self.F_max = 0.4, 0.9
        self.CR_min, self.CR_max = 0.7, 1.0
        self.diversity_threshold = 0.2
        self.adaptive_factor = 0.1
        self.history = []

    def compute_diversity(self, population):
        center = np.mean(population, axis=0)
        diversity = np.mean(np.linalg.norm(population - center, axis=1))
        return diversity

    def adapt_strategy(self, diversity):
        if diversity < self.diversity_threshold:
            return 1  # Exploitation phase
        else:
            return 0  # Exploration phase

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_quantum = np.random.uniform(0, 1, (self.population_size, self.dim))
        pop = lb + (ub - lb) * np.abs(np.sin(np.pi * population_quantum / 2))
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_pop = np.zeros_like(pop)
            diversity = self.compute_diversity(pop)
            strategy = self.adapt_strategy(diversity)

            for i in range(self.population_size):
                indices = np.random.choice(range(self.population_size), 3, replace=False)
                x0, x1, x2 = pop[indices]
                
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)

                if strategy == 0:
                    noise = np.random.uniform(-self.adaptive_factor, self.adaptive_factor, self.dim)
                    mutant = x0 + F * (x1 - x2) + noise
                else:
                    j_rand = np.random.randint(0, self.dim)
                    mutant = np.copy(best_global)
                    for j in range(self.dim):
                        if np.random.rand() < CR or j == j_rand:
                            mutant[j] = x0[j] + F * (best_global[j] - x0[j])

                mutant = np.clip(mutant, lb, ub)

                trial_fitness = func(mutant)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    next_pop[i] = mutant
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = mutant
                else:
                    next_pop[i] = pop[i]

            self.history.append(best_global)
            pop = next_pop

        return best_global