import numpy as np

class QuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.q_population = np.array([0.5 * np.ones(self.dim), 0.5 * np.ones(self.dim)])
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7

    def qbit_to_solution(self):
        return np.random.rand(self.population_size, self.dim) < self.q_population[0]

    def evaluate_population(self, func, population):
        return np.array([func(indiv) for indiv in population])

    def update_q_population(self, best_indiv):
        adaptation_factor = np.random.rand() * self.mutation_rate  # Adaptation for dynamic environments
        for i in range(self.dim):
            if best_indiv[i]:
                self.q_population[:, i] += adaptation_factor
            else:
                self.q_population[:, i] -= adaptation_factor
            self.q_population[:, i] = np.clip(self.q_population[:, i], 0, 1)

    def differential_evolution(self, population, scores):
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability
        new_population = np.copy(population)
        for i in range(self.population_size):
            candidates = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[candidates]
            mutant = np.clip(x1 + F * (x2 - x3), 0, 1)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            if func(trial) < scores[i]:
                new_population[i] = trial
        return new_population

    def __call__(self, func):
        current_eval = 0
        best_solution = None
        best_score = float('inf')

        while current_eval < self.budget:
            binary_population = self.qbit_to_solution()
            population = binary_population * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
            scores = self.evaluate_population(func, population)
            current_eval += len(scores)

            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_solution = population[min_idx]
            
            sorted_indices = np.argsort(scores)
            elite_individual = binary_population[min_idx]
            self.update_q_population(elite_individual)
            
            binary_population = self.differential_evolution(binary_population, scores)

            self.q_population += self.mutation_rate * np.random.uniform(-1, 1, self.q_population.shape)

        return best_solution