import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.5
        CR_initial = 0.9
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_dynamic = F * (1 + (0.5 * (evaluations / self.budget)))
                mutant = population[a] + F_dynamic * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                # Adaptive Crossover Rate
                CR_dynamic = CR_initial * (1 - (evaluations / self.budget)) + 0.1
                crossover = np.random.rand(self.dim) < CR_dynamic
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    # Diversity Preservation Mechanism
                    if np.random.rand() < 0.1:
                        random_index = np.random.randint(0, population_size)
                        random_candidate = np.random.uniform(lb, ub, self.dim)
                        random_fitness = func(random_candidate)
                        evaluations += 1
                        if random_fitness < fitness[random_index]:
                            population[random_index] = random_candidate
                            fitness[random_index] = random_fitness

                if np.random.rand() < 0.15:
                    local_search_point = population[i]
                    perturbation = np.random.normal(0, 0.1, self.dim) * (ub - lb)
                    neighbor = local_search_point + perturbation
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fitness = func(neighbor)
                    evaluations += 1

                    if neighbor_fitness < fitness[i]:
                        population[i] = neighbor
                        fitness[i] = neighbor_fitness

                if evaluations >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]