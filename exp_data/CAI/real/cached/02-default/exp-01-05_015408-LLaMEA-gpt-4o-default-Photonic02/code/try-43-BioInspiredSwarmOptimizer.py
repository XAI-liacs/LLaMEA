import numpy as np

class BioInspiredSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.pheromone_trail = np.ones((self.population_size, self.dim))
        self.evaporation_rate = 0.1
        self.learning_rate = 0.01

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position_population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        global_best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Generate candidate solution based on pheromone trail and local position adaptation
                candidate_position = self.generate_position(i, global_best_position, lb, ub)
                new_fitness = func(candidate_position)
                evaluations += 1

                # Update pheromone trails and learning
                self.update_pheromone(i, candidate_position, new_fitness < fitness[i])

                # Selection: keep the better solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    position_population[i] = candidate_position

                # Update global best position
                if new_fitness < fitness[best_index]:
                    best_index = i
                    global_best_position = candidate_position

                if evaluations >= self.budget:
                    break

        return global_best_position, fitness[best_index]

    def generate_position(self, i, global_best_position, lb, ub):
        # Generate new position using pheromone trail influence and adaptive learning
        direction = np.random.choice([-1, 1], self.dim)
        influence = self.pheromone_trail[i]
        candidate = influence * direction * (global_best_position - self.learning_rate * influence)
        candidate_position = np.clip(candidate, lb, ub) 
        return candidate_position

    def update_pheromone(self, i, candidate_position, is_improved):
        # Update pheromone trail based on solution improvement
        if is_improved:
            self.pheromone_trail[i] = (1 - self.evaporation_rate) * self.pheromone_trail[i] + self.evaporation_rate * candidate_position
        else:
            self.pheromone_trail[i] *= (1 - self.evaporation_rate)