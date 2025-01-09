import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20  # Population size for DE
        self.F = 0.8  # Differential weight factor
        self.CR = 0.9  # Crossover probability
        self.T0 = 100  # Initial temperature for SA
        self.cooling_rate = 0.99  # Cooling down rate for SA

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _mutate(self, pop, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = pop[a] + self.F * (pop[b] - pop[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _anneal(self, current_best, new_candidate, current_temp):
        delta = new_candidate - current_best
        if delta < 0 or np.random.rand() < np.exp(-delta / current_temp):
            return True
        return False

    def __call__(self, func):
        pop = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, pop)
        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]

        evals = self.pop_size
        temp = self.T0

        while evals < self.budget:
            for idx in range(self.pop_size):
                if evals >= self.budget:
                    break

                mutant = self._mutate(pop, idx)
                trial = self._crossover(pop[idx], mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[idx]:
                    pop[idx] = trial
                    fitness[idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                elif self._anneal(fitness[idx], trial_fitness, temp):
                    pop[idx] = trial
                    fitness[idx] = trial_fitness

            temp *= self.cooling_rate

        return best