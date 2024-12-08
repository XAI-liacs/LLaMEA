import numpy as np

class AdaptiveCrowdingDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.pop_size
        mutation_factor = np.ones(self.pop_size) * 0.5
        
        while evals < self.budget:
            crowding = np.zeros(self.pop_size)  # Initialize crowding distances
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                distances = np.linalg.norm(pop[idxs] - pop[i], axis=1)
                crowding[i] = np.sum(distances)  # Update crowding distance

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                crowding_factor = 1 + (np.max(crowding) - crowding[i]) / np.max(crowding)
                self.F = mutation_factor[i] * crowding_factor
                self.CR = 0.9 * crowding_factor  # Adjust crossover rate

                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.array([mutant[j] if np.random.rand() < self.CR else pop[i][j] for j in range(self.dim)])
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    mutation_factor[i] = min(mutation_factor[i] + 0.1, 1.0)
                else:
                    mutation_factor[i] = max(mutation_factor[i] - 0.1, 0.1)

                if evals >= self.budget:
                    break

                # Top-k random replacement for additional diversity
                if np.random.rand() < 0.2:
                    k = int(0.1 * self.pop_size)
                    random_indices = np.random.choice(self.pop_size, k, replace=False)
                    replacement = pop[random_indices[np.argmin(fitness[random_indices])]]
                    pop[i] = replacement
                    fitness[i] = func(pop[i])
                    evals += 1

            if evals >= self.budget:
                break
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]