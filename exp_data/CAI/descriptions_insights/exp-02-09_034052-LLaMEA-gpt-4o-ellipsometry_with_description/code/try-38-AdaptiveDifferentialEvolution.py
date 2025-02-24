import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f = 0.5  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.population = []
        self.memory_f = []  # Memory for f values
        self.memory_cr = []  # Memory for cr values

    def initialize_population(self, lb, ub):
        for _ in range(self.population_size):
            individual = np.random.uniform(lb, ub, self.dim)
            score = float('inf')
            self.population.append((individual, score))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0
        best_solution = None
        best_score = float('inf')

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target_vector, target_score = self.population[i]
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                x1, _ = self.population[a]
                x2, _ = self.population[b]
                x3, _ = self.population[c]

                # Introduce dynamic adjustment based on diversity
                diversity = np.std([ind[0] for ind in self.population], axis=0).mean()
                p = np.random.uniform()
                weighted_vector = self.f * (x2 - x3) if p < diversity else self.f * (x3 - x2)
                mutant_vector = x1 + weighted_vector
                mutant_vector = np.clip(mutant_vector, lb, ub)

                trial_vector = np.empty(self.dim)
                j_rand = np.random.randint(self.dim)

                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = target_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < target_score:
                    self.population[i] = (trial_vector, trial_score)

                    if trial_score < best_score:
                        best_solution = trial_vector
                        best_score = trial_score
                        self.memory_f.append(self.f)  # Store successful f
                        self.memory_cr.append(self.cr)  # Store successful cr

                # Adaptive parameter control with chaotic maps
                z = 0.7 + 0.3 * np.cos(2 * np.pi * evaluations / self.budget)
                if evaluations % (self.budget // 10) == 0:
                    self.f = z * np.random.uniform(0.4, 0.9)
                    self.cr = z * np.random.uniform(0.5, 1.0)
                    if self.memory_f:
                        self.f = np.mean(self.memory_f[-5:])  # Recent successful f
                    if self.memory_cr:
                        self.cr = np.mean(self.memory_cr[-5:])  # Recent successful cr

        return best_solution