import numpy as np

class AdaptivePopulationGradientOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = max(2, self.budget // 50)
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += population_size
        
        while self.evaluations < self.budget:
            # Sort population by fitness
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            # Adaptive population reduction
            if len(population) > max(2, self.budget // 200):
                population = population[:len(population) // 2]
                fitness = fitness[:len(fitness) // 2]

            # Gradient sampling and generation of new candidates
            new_population = []
            for individual in population:
                gradient = self.estimate_gradient(func, individual, lb, ub)
                candidate = individual - 0.01 * gradient
                candidate = np.clip(candidate, lb, ub)
                new_population.append(candidate)
            new_fitness = np.apply_along_axis(func, 1, np.array(new_population))
            self.evaluations += len(new_population)

            # Merge and select best
            population = np.vstack((population, new_population))
            fitness = np.hstack((fitness, new_fitness))

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def estimate_gradient(self, func, solution, lb, ub, epsilon=1e-4):
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = epsilon
            upper_sol = np.clip(solution + perturb, lb, ub)
            lower_sol = np.clip(solution - perturb, lb, ub)
            gradient[i] = (func(upper_sol) - func(lower_sol)) / (2 * epsilon)
            self.evaluations += 2
        return gradient