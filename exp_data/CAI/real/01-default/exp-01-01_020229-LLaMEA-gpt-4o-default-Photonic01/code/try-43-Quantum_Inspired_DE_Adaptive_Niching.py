import numpy as np

class Quantum_Inspired_DE_Adaptive_Niching:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5  # Scaling factor for mutation
        self.CR = 0.9  # Crossover probability
        self.q_factor = 0.9
        self.adaptive_mutation_scale = 0.1
        self.niching_radius = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                current_fitness = func(trial)
                evaluations += 1

                if current_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = current_fitness

                    if current_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = current_fitness

                # Quantum-Inspired Position Update
                if np.random.rand() < self.q_factor:
                    quantum_shift = np.random.normal(scale=self.adaptive_mutation_scale * (1 - evaluations / self.budget), size=self.dim)
                    quantum_position = np.clip(population[i] + quantum_shift, lb, ub)
                    quantum_fitness = func(quantum_position)
                    evaluations += 1

                    if quantum_fitness < fitness[i]:
                        population[i] = quantum_position
                        fitness[i] = quantum_fitness

                        if quantum_fitness < best_fitness:
                            best_individual = quantum_position
                            best_fitness = quantum_fitness
                
                if evaluations >= self.budget:
                    break

            # Niching Strategy to Maintain Diversity
            for i in range(self.population_size):
                distances = np.linalg.norm(population - population[i], axis=1)
                similar_indices = np.where(distances < self.niching_radius)[0]
                if len(similar_indices) > 1:
                    for idx in similar_indices:
                        if idx != i and fitness[idx] >= fitness[i]:
                            population[idx] = np.random.uniform(lb, ub, self.dim)
                            fitness[idx] = func(population[idx])
                            evaluations += 1

                            if fitness[idx] < best_fitness:
                                best_individual = population[idx]
                                best_fitness = fitness[idx]

                            if evaluations >= self.budget:
                                break

        return best_individual, best_fitness