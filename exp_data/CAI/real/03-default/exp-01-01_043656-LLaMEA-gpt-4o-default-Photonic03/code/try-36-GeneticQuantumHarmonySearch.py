import numpy as np

class GeneticQuantumHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.hmcr = 0.9  # Harmony Memory Consideration Rate
        self.par = 0.3   # Pitch Adjustment Rate
        self.quantum_factor_initial = 0.2
        self.quantum_factor_final = 0.05
        self.mutation_rate = 0.1

    def quantum_harmony_update(self, harmony, global_best, eval_count):
        lambda_factor = eval_count / self.budget
        quantum_factor = self.quantum_factor_initial * (1 - lambda_factor) + self.quantum_factor_final * lambda_factor
        delta = np.random.rand(self.dim)
        new_harmony = harmony + quantum_factor * (global_best - harmony) * delta
        return new_harmony

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        pop = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in pop])
        eval_count = self.population_size
        global_best = pop[np.argmin(fitness)]
        global_best_value = fitness.min()

        while eval_count < self.budget:
            new_pop = []
            for _ in range(self.population_size):
                if np.random.rand() < self.hmcr:
                    harmony_idx = np.random.choice(self.population_size)
                    new_harmony = pop[harmony_idx].copy()
                    if np.random.rand() < self.par:
                        new_harmony += np.random.normal(0, 0.1, self.dim)
                else:
                    new_harmony = np.random.rand(self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
                
                new_harmony = self.quantum_harmony_update(new_harmony, global_best, eval_count)
                new_harmony = np.clip(new_harmony, bounds[:, 0], bounds[:, 1])

                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.normal(0, 0.1, self.dim)
                    new_harmony = np.clip(new_harmony + mutation, bounds[:, 0], bounds[:, 1])

                new_pop.append(new_harmony)

            new_fitness = np.array([func(ind) for ind in new_pop])
            eval_count += self.population_size

            pop = np.array(new_pop)
            fitness = new_fitness
            if new_fitness.min() < global_best_value:
                global_best = new_pop[np.argmin(new_fitness)]
                global_best_value = new_fitness.min()

        return global_best