import numpy as np

class ModifiedDynamicInertiaHybridGA_PSO:
    def __init__(self, budget, dim, population_size=50, mutation_rate=0.1, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.chaos_param = 0.1  # Chaos parameter for chaotic map perturbation

    def chaotic_map_perturbation(self, population, chaos_param):
        return 2 * chaos_param * (population - np.floor(population))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        global_best = population[best_idx]
        velocity = np.zeros((self.population_size, self.dim))  # Initialize velocity

        mutation_rate_adjustment = 0.1
        for t in range(1, self.budget+1):
            # Update velocity based on PSO with chaotic map perturbation
            dynamic_inertia = self.inertia_weight * (1 - t/self.budget)  # Dynamic inertia weight
            velocity = dynamic_inertia * velocity + self.cognitive_weight * np.random.rand() * (best_individual - population) + self.social_weight * np.random.rand() * (global_best - population)
            velocity += self.chaotic_map_perturbation(velocity, self.chaos_param)  # Integrate chaotic map perturbation
            population += velocity

            # Dynamically adjust mutation rate
            mutation_rate_adjustment = 0.1 * np.exp(-t / self.budget)  # Exponential decay of mutation rate
            mutation_mask = np.random.rand(self.population_size, self.dim) < mutation_rate_adjustment
            population = population + np.random.uniform(-1.0, 1.0, (self.population_size, self.dim)) * mutation_mask

            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]
            global_best = population[np.argmin(fitness)]

        return global_best