import numpy as np
import scipy.stats

class EnhancedAdaptiveDynamicScalingWithQuantumMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * dim, 100)
        self.CR = 0.8  # Adjusted for dynamic exploitation
        self.F = 0.9  # Adjusted for dynamic exploration
        self.current_evaluations = 0

    def chaotic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def levy_flight(self, scale=0.015):  # Adjusted scale for better exploration
        return scipy.stats.levy_stable.rvs(alpha=1.5, beta=0, scale=scale, size=self.dim)

    def quantum_initialize_population(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        mean = (upper + lower) / 2
        std = (upper - lower) / 4  # Quantum-inspired spread
        return np.random.normal(mean, std, (self.population_size, self.dim))

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def mutate(self, target_idx, bounds, population, diversity_factor):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        chaotic_factor = 1 + 0.3 * self.chaotic_map(np.random.rand())  # Further chaotic influence
        adaptive_F = self.F * diversity_factor * chaotic_factor
        adaptive_F *= np.random.uniform(0.85, 1.6)  # Broadened dynamic scaling range
        mutant_vector = population[a] + adaptive_F * (population[b] - population[c])
        
        levy_step = self.levy_flight(scale=0.02 * diversity_factor)
        mutant_vector += levy_step if np.random.rand() > 0.3 else 0  # Conditional phase tweak
        
        opponent_idx = np.random.randint(self.population_size)
        opponent_vector = population[opponent_idx]
        mutation_scale = 0.05 * (1.0 - diversity_factor)  # Adjusted mutation
        gaussian_step = np.random.normal(0, mutation_scale, size=self.dim)
        mutant_vector += 0.5 * (opponent_vector - mutant_vector) + gaussian_step  # Increased opponent impact
        mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant, diversity_factor):
        adaptive_CR = self.CR * (1.0 - diversity_factor) * np.random.uniform(0.9, 1.2)  # Dynamic CR range
        crossover_points = np.random.rand(self.dim) < adaptive_CR
        trial_vector = np.where(crossover_points, mutant, target)
        return trial_vector

    def optimize(self, func, bounds):
        chaotic_sequence = np.random.rand(self.population_size)
        population = self.quantum_initialize_population(bounds)  # Quantum-inspired initialization
        fitness = self.evaluate(func, population)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            diversity = np.std(population, axis=0).mean()
            diversity_factor = 1.0 / (1.0 + np.sqrt(diversity))

            for i in range(self.population_size):
                target = population[i]
                mutant = self.mutate(i, bounds, population, diversity_factor)
                trial = self.crossover(target, mutant, diversity_factor)

                chaotic_sequence[i] = self.chaotic_map(chaotic_sequence[i])
                trial += chaotic_sequence[i] * (bounds.ub - bounds.lb) * np.random.uniform(0.01, 0.02)  # Adjusted scaling
                trial = np.clip(trial, bounds.lb, bounds.ub)

                trial_fitness = func(trial)
                self.current_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(target)
                    new_fitness.append(fitness[i])

                if self.current_evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array(new_fitness)

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        bounds = func.bounds
        best_solution, best_value = self.optimize(func, bounds)
        return best_solution, best_value