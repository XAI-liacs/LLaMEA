import numpy as np

class ESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.mutation_rate = 0.1
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            fitness = float('inf')
            population.append({'position': position, 'fitness': fitness})
        return population

    def evaluate_population(self, func):
        for individual in self.population:
            individual['fitness'] = func(individual['position'])

    def select_best(self):
        best = min(self.population, key=lambda ind: ind['fitness'])
        return best['position'].copy(), best['fitness']

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1
            individual['position'] += mutation_vector
            individual['position'] = np.clip(individual['position'], lb, ub)

    def anneal(self, candidate, best, lb, ub):
        candidate_energy = func(candidate)
        best_energy = func(best)
        if candidate_energy < best_energy or np.random.rand() < np.exp((best_energy - candidate_energy) / self.temperature):
            return candidate, candidate_energy
        return best, best_energy

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            self.evaluate_population(func)
            best_position, best_fitness = self.select_best()

            new_population = []
            for individual in self.population:
                self.mutate(individual, lb, ub)

                candidate, candidate_energy = self.anneal(individual['position'], best_position, lb, ub)
                evaluations += 1

                new_population.append({'position': candidate, 'fitness': candidate_energy})

                if evaluations >= self.budget:
                    break

            self.population = new_population
            self.temperature *= self.cooling_rate

            if evaluations >= self.budget:
                break

        return best_position, best_fitness