import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # angle of quantum rotation
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize quantum bits with equal probability of being 0 or 1
        q_population = np.random.rand(self.population_size, self.dim, 2)
        q_population /= np.linalg.norm(q_population, axis=2, keepdims=True)

        pop = self.measure(q_population, lb, ub)
        fitness = np.array([func(x) for x in pop])
        best_idx = np.argmin(fitness)
        best_global = pop[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            next_q_population = np.zeros_like(q_population)
            next_pop = np.zeros_like(pop)

            for i in range(self.population_size):
                # Quantum rotation gate mechanism
                delta_theta = self.alpha * (np.random.rand(self.dim) - 0.5)
                rotation_matrix = np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                                            [np.sin(delta_theta), np.cos(delta_theta)]])

                # Update quantum bit representation
                rotated_q_bits = q_population[i] @ rotation_matrix
                next_q_population[i] = rotated_q_bits / np.linalg.norm(rotated_q_bits, axis=1, keepdims=True)

                # Measure to get candidate solution
                candidate = self.measure(next_q_population[i][np.newaxis, ...], lb, ub).flatten()
                candidate_fitness = func(candidate)
                evaluations += 1

                if candidate_fitness < fitness[i]:
                    next_pop[i] = candidate
                    fitness[i] = candidate
                    if candidate_fitness < fitness[best_idx]:
                        best_idx = i
                        best_global = candidate
                else:
                    next_pop[i] = pop[i]

            q_population = next_q_population
            pop = next_pop
            self.history.append(best_global)

        return best_global

    def measure(self, q_population, lb, ub):
        # Measure quantum bits to obtain classical solutions
        binary_representation = np.random.rand(*q_population.shape[:-1]) < q_population[..., 0]
        return lb + (ub - lb) * (binary_representation / (2**np.arange(self.dim)))
