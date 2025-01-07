import numpy as np

class QuantumAnnealedGeneticSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 2 * dim)
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.inertia = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.annealing_rate = 0.98
        self.initial_temperature = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(individual) for individual in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.population_size
        temperature = self.initial_temperature

        while evaluations < self.budget:
            new_population = population.copy()
            
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    # Perform crossover
                    partner_index = np.random.randint(self.population_size)
                    point = np.random.randint(1, self.dim)
                    new_population[i, :point] = population[i, :point]
                    new_population[i, point:] = population[partner_index, point:]

                # Quantum annealing-inspired mutation
                for j in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        new_population[i, j] += np.random.normal(0, temperature) * (ub[j] - lb[j])
                        new_population[i, j] = np.clip(new_population[i, j], lb[j], ub[j])

            # Evaluate the new population
            new_scores = np.array([func(individual) for individual in new_population])
            evaluations += self.population_size

            # Select the best individuals
            for i in range(self.population_size):
                if new_scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = new_scores[i]
                    personal_best_positions[i] = new_population[i].copy()

                if new_scores[i] < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = new_population[i].copy()

            # Update population and velocities using swarm dynamics
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (self.inertia * velocities +
                          self.cognitive_coef * r1 * (personal_best_positions - population) +
                          self.social_coef * r2 * (global_best_position - population))
            population = np.clip(new_population + velocities, lb, ub)

            # Cool down the temperature
            temperature *= self.annealing_rate

        return global_best_position, personal_best_scores[global_best_index]