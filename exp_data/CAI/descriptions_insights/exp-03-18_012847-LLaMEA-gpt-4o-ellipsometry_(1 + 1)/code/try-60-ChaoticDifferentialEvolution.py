import numpy as np

class ChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.crossover_rate = 0.85
        self.F = 0.5
        self.lb, self.ub = None, None
        self.chaotic_seed = 0.5678

    def chaotic_sequence(self, seed, n):
        x = seed
        result = []
        for _ in range(n):
            x = 3.91 * x * (1.0 - x)
            result.append(x)
        return result

    def initialize_population(self):
        chaotic_seq = self.chaotic_sequence(self.chaotic_seed, self.population_size * self.dim)
        population = np.array(chaotic_seq).reshape(self.population_size, self.dim)
        return self.lb + population * (self.ub - self.lb)

    def adaptive_scaling_factor(self, success_rate):
        return self.F * (0.9 + 0.1 * np.tanh(success_rate))

    def adaptive_crossover_rate(self, iteration, max_iter):
        return 0.7 + 0.3 * np.sin(np.pi * iteration / max_iter)

    def adaptive_chaotic_weight(self, iteration):  # New function added
        return 0.3 + 0.7 * np.sin(2 * np.pi * iteration / self.population_size)

    def tournament_selection(self, fitness):  # New function added
        idx1, idx2 = np.random.choice(len(fitness), 2, replace=False)
        return idx1 if fitness[idx1] < fitness[idx2] else idx2

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        evaluations = self.population_size
        adaptive_F = self.F

        max_iter = self.budget // self.population_size
        for iteration in range(max_iter):
            success_count = 0
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                d = self.tournament_selection(fitness)  # Modified line
                chaotic_weight = self.adaptive_chaotic_weight(iteration)  # Modified line
                mutant = best_solution + adaptive_F * (population[b] - population[c]) + chaotic_weight * (population[d] - population[i])  # Modified line
                mutant = np.clip(mutant, self.lb, self.ub)

                self.crossover_rate = self.adaptive_crossover_rate(iteration, max_iter)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    success_count += 1

                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial
                        adaptive_F = self.adaptive_scaling_factor(success_count / self.population_size)

                if evaluations >= self.budget:
                    break

        return best_solution