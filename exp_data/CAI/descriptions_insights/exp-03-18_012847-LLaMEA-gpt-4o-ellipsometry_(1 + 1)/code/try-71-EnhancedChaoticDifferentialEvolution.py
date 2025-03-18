import numpy as np

class EnhancedChaoticDifferentialEvolution:
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

    def chaotic_mutation(self, candidate, best_solution, b, c, d):
        chaotic_factor = np.random.uniform(0, 1)
        return candidate + chaotic_factor * (best_solution - candidate) + self.F * (b - c) + chaotic_factor * (d - candidate)
    
    def self_adaptive_population(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[:len(population) // 2]]

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
                d = np.random.choice(indices)
                
                mutant = self.chaotic_mutation(population[i], best_solution, population[b], population[c], population[d])
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

            population = self.self_adaptive_population(population, fitness)
            self.population_size = len(population)

        return best_solution