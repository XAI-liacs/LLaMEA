import numpy as np

class AdaptiveQuantumEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = np.random.rand(self.population_size, dim)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.8  # Mutation factor for differential evolution
        self.crossover_rate = 0.7  # Crossover rate for differential evolution

    def measure(self, q_individual):
        return np.round(q_individual)

    def evaluate_population(self, func):
        classical_population = np.array([self.measure(q_ind) for q_ind in self.q_population])
        fitness = np.array([func(ind) for ind in classical_population])
        return fitness, classical_population

    def differential_evolution(self, index, classical_population):
        indices = [i for i in range(self.population_size) if i != index]
        a, b, c = classical_population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = a + self.mutation_factor * (b - c)
        crossover = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover, mutant_vector, classical_population[index])
        return np.clip(trial_vector, 0, 1)

    def update_quantum_population(self, fitness, classical_population):
        sorted_indices = np.argsort(fitness)
        best_indices = sorted_indices[:self.population_size // 2]
        for i in range(self.population_size):
            if i not in best_indices:
                best_individual = classical_population[np.random.choice(best_indices)]
                self.q_population[i] += 0.4 * (best_individual - self.q_population[i])
                self.q_population[i] = np.clip(self.q_population[i], 0, 1)
                
                # Differential evolution step
                trial_vector = self.differential_evolution(i, classical_population)
                trial_fitness = func(trial_vector)
                if trial_fitness < fitness[i]:
                    classical_population[i] = trial_vector
                    fitness[i] = trial_fitness

        random_indices = np.random.choice(self.population_size, self.population_size // 10, replace=False)
        self.q_population[random_indices] = np.random.rand(len(random_indices), self.dim)

    def __call__(self, func):
        func_bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        evaluations = 0

        while evaluations < self.budget:
            fitness, classical_population = self.evaluate_population(func)
            evaluations += len(fitness)

            min_fitness_index = np.argmin(fitness)
            if fitness[min_fitness_index] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_index]
                self.best_solution = classical_population[min_fitness_index]

            if evaluations >= self.budget:
                break

            self.update_quantum_population(fitness, classical_population)

        real_solution = self.best_solution * (func_bounds[:, 1] - func_bounds[:, 0]) + func_bounds[:, 0]
        return real_solution