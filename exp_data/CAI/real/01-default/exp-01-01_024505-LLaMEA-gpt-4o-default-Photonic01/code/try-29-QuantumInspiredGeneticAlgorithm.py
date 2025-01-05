import numpy as np

class QuantumInspiredGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = max(20, min(100, budget // 5))
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_rate = 0.1
        self.entanglement_factor = 0.6
        self.generational_progress = 0
        self.learning_rate = 0.1
        self.exploration_phase = True
        self.diversity_threshold = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.initial_pop_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        best_index = np.argmin(fitness)
        if fitness[best_index] < self.best_fitness:
            self.best_fitness = fitness[best_index]
            self.best_solution = self.population[best_index]
        return fitness

    def select_parents(self, fitness):
        selection_prob = 1 / (fitness + 1e-9)
        selection_prob /= selection_prob.sum()
        indices = np.random.choice(len(fitness), size=len(fitness), p=selection_prob)
        return self.population[indices]

    def crossover(self, parents):
        offspring = np.empty_like(parents)
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):
                break
            alpha = np.random.rand(self.dim)
            offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i+1]
            offspring[i+1] = alpha * parents[i+1] + (1 - alpha) * parents[i]
        return offspring

    def mutate(self, offspring, lb, ub):
        mutation_matrix = np.random.rand(*offspring.shape) < self.mutation_rate
        random_values = lb + (ub - lb) * np.random.rand(*offspring.shape)
        offspring = np.where(mutation_matrix, random_values, offspring)
        return np.clip(offspring, lb, ub)

    def apply_quantum_entanglement(self, offspring):
        for i in range(len(offspring)):
            if np.random.rand() < self.entanglement_factor:
                partner_idx = np.random.randint(len(offspring))
                qubit_superposition = 0.5 * (offspring[i] + offspring[partner_idx])
                offspring[i] = qubit_superposition + np.random.normal(0, self.learning_rate, self.dim)
        return offspring

    def quantum_tunneling(self, offspring, lb, ub):
        diversity = np.std(offspring, axis=0).mean()
        if diversity < self.diversity_threshold:
            indices = np.random.choice(len(offspring), size=3, replace=False)
            random_individuals = lb + (ub - lb) * np.random.rand(3, self.dim)
            offspring[indices] = random_individuals
        return np.clip(offspring, lb, ub)

    def adapt_population_size(self):
        if self.generational_progress > self.budget // 3:
            self.exploration_phase = False
        if self.exploration_phase:
            new_pop_size = int(self.initial_pop_size * 1.2)
        else:
            new_pop_size = max(int(self.initial_pop_size * 0.8), 10)
        self.population = self.population[:new_pop_size]

    def update_mutation_rate_and_learning_rate(self):
        self.mutation_rate = max(0.01, 0.1 * (1 - self.generational_progress / self.budget))
        self.learning_rate = max(0.01, 0.1 * (1 - self.best_fitness / (self.best_fitness + 1e-9)))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += len(fitness)

            if evaluations >= self.budget:
                break

            parents = self.select_parents(fitness)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring, lb, ub)
            offspring = self.apply_quantum_entanglement(offspring)
            offspring = self.quantum_tunneling(offspring, lb, ub)

            self.population = offspring
            self.adapt_population_size()
            self.update_mutation_rate_and_learning_rate()
            self.generational_progress = evaluations

        return self.best_solution, self.best_fitness