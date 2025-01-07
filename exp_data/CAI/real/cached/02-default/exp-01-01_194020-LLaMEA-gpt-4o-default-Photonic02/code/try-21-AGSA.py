import numpy as np

class AGSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.temperature = 100.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            individual = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': individual, 'value': float('inf')})
        return population

    def mutate(self, individual, lb, ub):
        mutation_strength = 0.1 * (ub - lb)
        mutated_position = individual['position'] + mutation_strength * np.random.randn(self.dim)
        mutated_position = np.clip(mutated_position, lb, ub)
        return mutated_position

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        offspring = alpha * parent1['position'] + (1 - alpha) * parent2['position']
        return offspring

    def simulated_annealing_acceptance(self, candidate_value, current_value):
        if candidate_value < current_value:
            return True
        else:
            return np.random.rand() < np.exp((current_value - candidate_value) / self.temperature)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            # Evaluate individuals
            for individual in population:
                if individual['value'] == float('inf'):
                    individual['value'] = func(individual['position'])
                    evaluations += 1
                    if individual['value'] < self.best_value:
                        self.best_value = individual['value']
                        self.best_solution = individual['position'].copy()

                if evaluations >= self.budget:
                    break

            # Create new generation
            new_population = []
            for _ in range(self.population_size // 2):
                parents = np.random.choice(population, 2, replace=False)
                offspring1_pos = self.crossover(parents[0], parents[1])
                offspring2_pos = self.crossover(parents[1], parents[0])
                
                for offspring_pos in [offspring1_pos, offspring2_pos]:
                    offspring_value = func(offspring_pos)
                    evaluations += 1
                    if offspring_value < self.best_value:
                        self.best_value = offspring_value
                        self.best_solution = offspring_pos.copy()

                    choice = parents[0] if self.simulated_annealing_acceptance(offspring_value, parents[0]['value']) else parents[1]
                    mutated_offspring_pos = self.mutate({'position': offspring_pos}, lb, ub)
                    mutated_offspring_value = func(mutated_offspring_pos)
                    evaluations += 1
                    if mutated_offspring_value < self.best_value:
                        self.best_value = mutated_offspring_value
                        self.best_solution = mutated_offspring_pos.copy()
                    
                    new_population.append({'position': mutated_offspring_pos, 'value': mutated_offspring_value})

                if evaluations >= self.budget:
                    break

            # Update population and temperature
            population = new_population
            self.temperature *= self.cooling_rate

        return self.best_solution, self.best_value