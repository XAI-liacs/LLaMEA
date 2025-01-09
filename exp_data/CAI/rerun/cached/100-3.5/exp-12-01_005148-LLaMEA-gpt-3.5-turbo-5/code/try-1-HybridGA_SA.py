import numpy as np

class HybridGA_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.sa_iterations = 100
        self.temperature_max = 10.0
        self.temperature_min = 1e-3

    def __call__(self, func):
        def simulate_annealing(current_solution, current_temperature):
            best_solution = current_solution
            best_fitness = func(best_solution)
            for _ in range(self.sa_iterations):
                new_solution = current_solution + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, -5.0, 5.0)
                new_fitness = func(new_solution)
                if new_fitness < best_fitness or np.exp((best_fitness - new_fitness) / current_temperature) > np.random.rand():
                    best_solution = new_solution
                    best_fitness = new_fitness
                current_temperature *= 0.9
                current_temperature = max(current_temperature, self.temperature_min)
            return best_solution

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        for _ in range(self.budget // self.population_size):
            children = []
            for _ in range(self.population_size):
                parent1 = population[np.random.randint(0, self.population_size)]
                parent2 = population[np.random.randint(0, self.population_size)]
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                child = np.where(crossover_mask, parent1, parent2)
                mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                child += np.random.normal(0, 0.1, self.dim) * mutation_mask
                child = np.clip(child, -5.0, 5.0)
                child = simulate_annealing(child, self.temperature_max)
                children.append(child)
            population = np.array(children)

        best_solution = population[np.argmin([func(sol) for sol in population])]
        return best_solution