import numpy as np
import scipy.stats

class EnhancedHybridQuantumGaussianWalkWithMultiLayeredLandscapeLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(10 * dim, 100)
        self.CR = 0.8  # Adjusted crossover probability
        self.F = 0.9  # Adjusted mutation factor
        self.current_evaluations = 0

    def quantum_gaussian_walk(self, bounds):
        lower, upper = bounds.lb, bounds.ub
        mean = (upper + lower) / 2
        std = (upper - lower) / 5
        init_population = np.random.normal(mean, std, (self.population_size, self.dim))
        perturb_scale = 0.03 * (1 + np.random.rand())
        perturb = np.random.normal(0, std * perturb_scale, (self.population_size, self.dim))  # Adaptive perturbation scale
        return np.clip(init_population + perturb, lower, upper)

    def evaluate(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.current_evaluations += len(population)
        return fitness

    def mutate(self, target_idx, bounds, population, landscape_factor):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), target_idx), 3, replace=False)
        adaptive_F = self.F * (landscape_factor ** 1.2)
        mutant_vector = population[a] + adaptive_F * (population[b] - population[c])
        
        opponent_idx = np.random.randint(self.population_size)
        opponent_vector = population[opponent_idx]
        mutation_scale = 0.08 * (1.0 - landscape_factor) * (1 + np.random.rand())  # Adaptive mutation scale
        gaussian_step = np.random.normal(0, mutation_scale, size=self.dim)
        mutant_vector += 0.4 * (opponent_vector - mutant_vector) + gaussian_step
        mutant_vector = np.clip(mutant_vector, bounds.lb, bounds.ub)
        return mutant_vector

    def crossover(self, target, mutant, landscape_factor):
        adaptive_CR = self.CR * (1.0 - landscape_factor ** 0.5)  # Adjusted CR adaptation
        crossover_points = np.random.rand(self.dim) < adaptive_CR
        trial_vector = np.where(crossover_points, mutant, target)
        return trial_vector

    def optimize(self, func, bounds):
        population = self.quantum_gaussian_walk(bounds)
        fitness = self.evaluate(func, population)

        while self.current_evaluations < self.budget:
            new_population = []
            new_fitness = []

            diversity = np.std(population, axis=0).mean()
            landscape_factor = 1.0 / (1.0 + 2 * np.sqrt(diversity))  # Adjusted landscape learning
            
            for i in range(self.population_size):
                target = population[i]
                mutant = self.mutate(i, bounds, population, landscape_factor)
                trial = self.crossover(target, mutant, landscape_factor)
                
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