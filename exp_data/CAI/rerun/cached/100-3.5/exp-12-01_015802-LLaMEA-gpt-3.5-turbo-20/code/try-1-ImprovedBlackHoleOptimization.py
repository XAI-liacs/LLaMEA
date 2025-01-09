import numpy as np

class ImprovedBlackHoleOptimization(BlackHoleOptimization):
    def __init__(self, budget, dim, num_holes=10, num_iterations=100, inertia_weight=0.5):
        super().__init__(budget, dim, num_holes, num_iterations)
        self.inertia_weight = inertia_weight

    def __call__(self, func):
        def update_position(population, fitness):
            best_idx = np.argmax(fitness)
            centroid = np.mean(population, axis=0)
            new_population = population.copy()
            for i in range(self.num_holes):
                if i != best_idx:
                    direction = population[i] - population[best_idx]
                    distance = np.linalg.norm(population[i] - population[best_idx])
                    inertia_term = self.inertia_weight * np.random.uniform() * direction / distance
                    social_term = centroid - population[i]
                    new_population[i] = population[i] + inertia_term + social_term
            return new_population

        population = initialize_population()
        fitness = calculate_fitness(population)
        
        for _ in range(self.num_iterations):
            new_population = update_position(population, fitness)
            new_fitness = calculate_fitness(new_population)
            if np.max(new_fitness) > np.max(fitness):
                population = new_population
                fitness = new_fitness

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        
        return best_solution