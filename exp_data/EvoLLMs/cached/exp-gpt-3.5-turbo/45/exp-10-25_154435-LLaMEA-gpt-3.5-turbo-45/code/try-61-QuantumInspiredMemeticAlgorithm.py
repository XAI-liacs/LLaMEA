import numpy as np

class QuantumInspiredMemeticAlgorithm:
    def __init__(self, budget, dim, num_particles=30, num_iterations=1000, alpha=0.2, beta=0.5, local_search_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.local_search_prob = local_search_prob

    def quantum_rotation_gate(self, population):
        rotation_angle = np.random.uniform(0, 2*np.pi, size=(self.dim, self.num_particles))
        population = np.transpose(population) * np.exp(1j * rotation_angle)
        return np.transpose(population)

    def local_search(self, individual, func):
        perturbed_individual = individual + np.random.normal(0, 0.1, size=self.dim)
        return perturbed_individual if func(perturbed_individual) < func(individual) else individual

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        for _ in range(self.num_iterations):
            population = self.quantum_rotation_gate(population)
            fitness_values = [func(individual) for individual in population]
            top_indices = np.argsort(fitness_values)[:self.num_particles//2]
            best_individual = population[np.argmin(fitness_values)]
            updated_population = self.alpha * best_individual + np.sqrt(1-self.alpha**2) * population

            for i, individual in enumerate(updated_population):
                if np.random.rand() < self.local_search_prob:
                    updated_population[i] = self.local_search(individual, func)

            updated_fitness_values = [func(individual) for individual in updated_population]
            population = updated_population if min(updated_fitness_values) < min(fitness_values) else population

        best_solution = population[np.argmin([func(individual) for individual in population])]
        return best_solution