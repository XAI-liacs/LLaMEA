import numpy as np

class DualPopulationCoEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.exploratory_population_size = 5 * dim
        self.exploitative_population_size = 5 * dim
        self.alpha = 0.3
        self.beta = 0.7

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize exploratory and exploitative populations
        exploratory_population = np.random.rand(self.exploratory_population_size, self.dim)
        exploitative_population = np.random.rand(self.exploitative_population_size, self.dim)

        pos_exploratory = self.quantum_to_position(exploratory_population, lb, ub)
        pos_exploitative = self.quantum_to_position(exploitative_population, lb, ub)

        fitness_exploratory = np.array([func(ind) for ind in pos_exploratory])
        fitness_exploitative = np.array([func(ind) for ind in pos_exploitative])

        evaluations = self.exploratory_population_size + self.exploitative_population_size
        best_index_exploratory = np.argmin(fitness_exploratory)
        best_index_exploitative = np.argmin(fitness_exploitative)

        best_overall = pos_exploratory[best_index_exploratory] if fitness_exploratory[best_index_exploratory] < fitness_exploitative[best_index_exploitative] else pos_exploitative[best_index_exploitative]

        while evaluations < self.budget:
            # Explore (divergent search)
            for i in range(self.exploratory_population_size):
                if np.random.rand() < self.alpha:
                    exploratory_population[i] = self.random_mutation(exploratory_population[i])
                pos_exploratory[i] = self.quantum_to_position(exploratory_population[i], lb, ub)
                new_fitness = func(pos_exploratory[i])
                evaluations += 1

                if new_fitness < fitness_exploratory[i]:
                    fitness_exploratory[i] = new_fitness

                if new_fitness < fitness_exploratory[best_index_exploratory]:
                    best_index_exploratory = i

                if evaluations >= self.budget:
                    break

            # Exploit (convergent search)
            for j in range(self.exploitative_population_size):
                if np.random.rand() < self.beta:
                    exploitative_population[j] = self.gradient_ascent(exploitative_population[j], pos_exploitative[best_index_exploitative])
                pos_exploitative[j] = self.quantum_to_position(exploitative_population[j], lb, ub)
                new_fitness = func(pos_exploitative[j])
                evaluations += 1

                if new_fitness < fitness_exploitative[j]:
                    fitness_exploitative[j] = new_fitness

                if new_fitness < fitness_exploitative[best_index_exploitative]:
                    best_index_exploitative = j

                if evaluations >= self.budget:
                    break

            # Synergistic update
            if fitness_exploratory[best_index_exploratory] < fitness_exploitative[best_index_exploitative]:
                best_overall = pos_exploratory[best_index_exploratory]
            else:
                best_overall = pos_exploitative[best_index_exploitative]

        return best_overall, min(fitness_exploratory[best_index_exploratory], fitness_exploitative[best_index_exploitative])

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def random_mutation(self, quantum_bits):
        mutation_factor = np.random.uniform(-0.5, 0.5, size=quantum_bits.shape)
        new_bits = quantum_bits + mutation_factor
        new_bits = np.clip(new_bits, 0, 1)
        return new_bits

    def gradient_ascent(self, quantum_bits, reference_bits):
        delta_theta = self.alpha * (reference_bits - quantum_bits)
        new_bits = quantum_bits + delta_theta
        new_bits = np.clip(new_bits, 0, 1)
        return new_bits