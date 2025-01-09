import numpy as np

class HybridFireflyGeneticAlgorithm:
    def __init__(self, budget, dim, num_fireflies=20, num_gen=100, p_crossover=0.8, p_mutation=0.1):
        self.budget = budget
        self.dim = dim
        self.num_fireflies = num_fireflies
        self.num_gen = num_gen
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.num_fireflies, self.dim))

        def firefly_move(curr_firefly, best_firefly):
            beta0 = 1.0
            gamma = 0.1
            alpha = 0.2
            epsilon = 0.01
            r = np.linalg.norm(curr_firefly - best_firefly)
            beta = beta0 * np.exp(-gamma * r**2)
            return curr_firefly + alpha * (best_firefly - curr_firefly) + epsilon * (np.random.rand(self.dim) - 0.5)

        def evaluate_population(population):
            return np.array([func(ind) for ind in population])

        def genetic_crossover(parent1, parent2):
            mask = np.random.randint(0, 2, size=self.dim)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2

        def genetic_mutation(individual):
            mask = np.random.rand(self.dim) < self.p_mutation
            individual[mask] = np.random.uniform(-5.0, 5.0, size=np.sum(mask))
            return individual

        population = initialize_population()
        fitness_values = evaluate_population(population)
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index].copy()

        for _ in range(self.num_gen):
            for i in range(self.num_fireflies):
                new_position = firefly_move(population[i], best_solution)
                if np.any(new_position < -5.0) or np.any(new_position > 5.0):
                    continue
                new_fitness = func(new_position)
                if new_fitness < fitness_values[i]:
                    population[i] = new_position
                    fitness_values[i] = new_fitness

            population = np.array([genetic_mutation(individual) for individual in population])

            for _ in range(self.num_fireflies // 2):
                idx1, idx2 = np.random.choice(self.num_fireflies, size=2, replace=False)
                if np.random.rand() < self.p_crossover:
                    population[idx1], population[idx2] = genetic_crossover(population[idx1], population[idx2])

            current_best_index = np.argmin(fitness_values)
            if fitness_values[current_best_index] < fitness_values[best_index]:
                best_index = current_best_index
                best_solution = population[best_index].copy()

        return best_solution