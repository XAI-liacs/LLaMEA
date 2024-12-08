import numpy as np

class MultiPopulationDEWithMigration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        max_ls_iter = 5
        num_populations = 5
        migration_interval = 10

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local):
            best_candidate = np.copy(candidate)
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
            return best_candidate

        def mutate(current, candidates, F, CR):
            mutated = np.copy(current)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    if np.random.rand() < 0.5:
                        F_i = F + 0.1 * np.random.randn()
                        F_i = max(0.1, min(0.9, F_i))  # Dynamic adjustment of F_i
                    else:
                        F_i = F
                    mutated[i] = target_to_bounds(candidates[a, i] + F_i * (candidates[b, i] - candidates[c, i]), -5.0, 5.0)
            return mutated

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        F = 0.8
        CR = 0.9

        populations = [np.random.uniform(-5.0, 5.0, (pop_size, self.dim)) for _ in range(num_populations)]
        for _ in range(self.budget // (pop_size * num_populations)):
            for p_idx in range(num_populations):
                population = populations[p_idx]
                for i in range(pop_size):
                    trial = mutate(population[i], population, F, CR)
                    mutated_fitness = func(trial)
                    if mutated_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = mutated_fitness
                    population[i] = local_search(population[i], func)
                
                if _ % migration_interval == 0 and p_idx < num_populations - 1:
                    populations[p_idx + 1] = population
        
        return best_solution