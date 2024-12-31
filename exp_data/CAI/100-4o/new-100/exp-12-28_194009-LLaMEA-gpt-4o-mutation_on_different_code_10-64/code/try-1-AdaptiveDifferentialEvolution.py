import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = max(20, 5 * dim)
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.scale_factors = np.random.uniform(0.5, 1.0, self.pop_size)
        self.crossover_probs = np.random.uniform(0.1, 0.9, self.pop_size)

    def mutation(self, idx, best_idx):
        candidates = [i for i in range(self.pop_size) if i != idx]
        a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
        return a + self.scale_factors[idx] * (b - c) + 0.5 * (self.population[best_idx] - self.population[idx])

    def crossover(self, target, mutant, cr):
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, mutant, target)

    def __call__(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        self.f_opt, self.x_opt = fitness[best_idx], self.population[best_idx]

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutation(i, best_idx)
                trial = self.crossover(self.population[i], mutant, self.crossover_probs[i])
                trial = np.clip(trial, -5.0, 5.0)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt, self.x_opt = trial_fitness, trial

                    self.scale_factors[i] = np.clip(self.scale_factors[i] * 1.2, 0.1, 1.0)
                    self.crossover_probs[i] = min(self.crossover_probs[i] + 0.1, 0.9)
                else:
                    self.scale_factors[i] = np.clip(self.scale_factors[i] * 0.8, 0.1, 1.0)
                    self.crossover_probs[i] = max(self.crossover_probs[i] - 0.1, 0.1)
                
                if len(fitness) > 10 and evaluations % (self.budget // 10) == 0:
                    worst_idx = np.argmax(fitness)
                    self.population = np.delete(self.population, worst_idx, axis=0)
                    fitness = np.delete(fitness, worst_idx)
                    self.pop_size -= 1

                if evaluations >= self.budget:
                    break

            best_idx = np.argmin(fitness)

        return self.f_opt, self.x_opt