import numpy as np

class HybridQuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5
        self.beta = 0.5
        self.adaptive_factor = 0.1
        self.niching_radius = 0.1 * (func.bounds.ub - func.bounds.lb)
        self.niche_capacity = int(0.1 * self.population_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                if np.random.rand() < self.alpha:
                    quantum_population[i] = self.update_quantum_bits(quantum_population[i], quantum_population[best_index], fitness, i)

                position_population[i] = self.quantum_to_position(quantum_population[i], lb, ub)
                new_fitness = func(position_population[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness

                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = position_population[i]

                if evaluations >= self.budget:
                    break

            self.dynamic_niching(quantum_population, position_population, fitness)

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position

    def update_quantum_bits(self, quantum_bits, best_quantum_bits, fitness, index):
        delta_theta = self.beta * (best_quantum_bits - quantum_bits)
        improvement_ratio = (fitness[index] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-9)
        adaptive_delta = delta_theta * (1 + self.adaptive_factor * (0.5 - improvement_ratio))
        new_quantum_bits = quantum_bits + adaptive_delta
        new_quantum_bits = np.clip(new_quantum_bits, 0, 1)
        return new_quantum_bits

    def dynamic_niching(self, quantum_population, position_population, fitness):
        niches = []
        for i, pos in enumerate(position_population):
            placed_in_niche = False
            for niche in niches:
                if np.linalg.norm(pos - niche['center']) < self.niching_radius:
                    if len(niche['members']) < self.niche_capacity:
                        niche['members'].append(i)
                    placed_in_niche = True
                    break

            if not placed_in_niche:
                niches.append({'center': pos, 'members': [i]})

        for niche in niches:
            niche_fitness = [fitness[i] for i in niche['members']]
            best_niche_index = niche['members'][np.argmin(niche_fitness)]
            quantum_population[niche['members']] = quantum_population[best_niche_index]