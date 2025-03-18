import numpy as np

class QuantumInspiredGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Population size for genetic algorithm
        self.quantum_size = 10  # Quantum-inspired solutions
        self.quantum_angle = np.pi / 4  # Quantum rotation angle
        self.mutation_rate = 0.1  # Mutation rate
        self.cross_prob = 0.7  # Crossover probability

    def initialize_population(self, lb, ub):
        # Initialize a classical population
        classical_pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        # Initialize a quantum-inspired population
        quantum_pop = np.random.uniform(0, 1, (self.quantum_size, self.dim))
        return classical_pop, quantum_pop

    def quantum_to_classical(self, quantum_pop, lb, ub):
        # Convert quantum-inspired individuals to classical domain
        return lb + (ub - lb) * (np.sin(quantum_pop * self.quantum_angle) ** 2)

    def evaluate_population(self, pop, func):
        return np.array([func(ind) for ind in pop])

    def select_parents(self, classical_pop, fitness):
        idx = np.random.choice(range(self.population_size), size=2, p=fitness / fitness.sum())
        return classical_pop[idx]

    def crossover(self, parents):
        if np.random.rand() < self.cross_prob:
            alpha = np.random.rand(self.dim) * 0.5 + 0.5  # Ensure more diversity
            return parents[0] * alpha + parents[1] * (1 - alpha)
        else:
            return parents[np.random.randint(0, 2)]

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.uniform(-0.2, 0.2, self.dim)  # Increase mutation range
            mutated = individual + mutation_vector
            return np.clip(mutated, lb, ub)
        else:
            return individual

    def update_quantum_population(self, quantum_pop, best_individual, lb, ub):
        classical_best = self.quantum_to_classical(quantum_pop, lb, ub)
        best_index = np.argmin(np.array([np.linalg.norm(ind - best_individual) for ind in classical_best]))
        quantum_pop[best_index] += np.random.normal(0, 0.1, self.dim)
        return np.clip(quantum_pop, 0, 1)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        classical_pop, quantum_pop = self.initialize_population(lb, ub)
        evaluations = 0
        best_solution = None
        best_fitness = float('inf')

        while evaluations < self.budget:
            # Convert and evaluate populations
            classical_solutions = classical_pop
            quantum_solutions = self.quantum_to_classical(quantum_pop, lb, ub)
            total_population = np.vstack((classical_solutions, quantum_solutions))
            fitness = self.evaluate_population(total_population, func)
            evaluations += len(total_population)

            # Find best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = total_population[min_idx]

            # Genetic operations
            classical_fitness = fitness[:self.population_size]
            new_classical_pop = []
            for _ in range(self.population_size):
                parents = self.select_parents(classical_pop, classical_fitness)
                offspring = self.crossover(parents)
                offspring = self.mutate(offspring, lb, ub)
                new_classical_pop.append(offspring)
            classical_pop = np.array(new_classical_pop)

            # Update quantum population
            quantum_pop = self.update_quantum_population(quantum_pop, best_solution, lb, ub)

        return best_solution