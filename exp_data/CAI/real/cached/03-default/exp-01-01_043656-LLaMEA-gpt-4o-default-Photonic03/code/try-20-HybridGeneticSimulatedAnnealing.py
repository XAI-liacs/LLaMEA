import numpy as np

class HybridGeneticSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 5 * dim)
        self.mutation_rate = 0.1
        self.initial_temperature = 1000
        self.final_temperature = 1
        self.alpha = 0.99

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, individual):
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        mutation_values = np.random.randn(self.dim) * mutation_mask
        return individual + mutation_values

    def simulated_annealing(self, individual, func, temperature):
        neighbor = self.mutate(individual)
        delta_energy = func(neighbor) - func(individual)
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            return neighbor
        return individual

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness = np.array([func(ind) for ind in population])
        best_individual = population[np.argmin(fitness)]
        best_fitness = fitness.min()

        eval_count = self.population_size

        temperature = self.initial_temperature
        while eval_count < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                selected_indices = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[selected_indices]
                offspring1 = self.crossover(parent1, parent2)
                offspring2 = self.crossover(parent2, parent1)

                offspring1 = np.clip(offspring1, bounds[:, 0], bounds[:, 1])
                offspring2 = np.clip(offspring2, bounds[:, 0], bounds[:, 1])

                offspring1 = self.simulated_annealing(offspring1, func, temperature)
                offspring2 = self.simulated_annealing(offspring2, func, temperature)

                new_population.extend([offspring1, offspring2])

                eval_count += 2
                if eval_count >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_individual = population[current_best_index]
                best_fitness = fitness[current_best_index]

            temperature *= self.alpha

        return best_individual