import numpy as np

class AdaptiveDifferentialLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(4 + int(3 * np.log(self.dim)), 50)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.inertia_weight = 0.5  # Added for swarm behavior
        self.cognitive_coef = 1.5  # Added for swarm behavior
        self.social_coef = 1.5     # Added for swarm behavior
        
    def __call__(self, func):
        bounds = func.bounds
        lower_bound, upper_bound = bounds.lb, bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))  # Added for PSO-like behavior
        fitness = np.array([func(ind) for ind in population])
        personal_best = population.copy()  # Added for tracking personal bests
        personal_best_fitness = fitness.copy()  # Added for tracking personal best fitness
        
        for _ in range(self.budget - self.population_size):
            global_best = population[np.argmin(fitness)]  # Added for swarm behavior
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)  # Added for random coefficients
                # Update velocity with PSO components
                velocity[i] = (self.inertia_weight * velocity[i]
                               + self.cognitive_coef * r1 * (personal_best[i] - population[i])
                               + self.social_coef * r2 * (global_best - population[i]))
                # Differential Mutation
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c) + velocity[i], lower_bound, upper_bound)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i], fitness[i] = trial, trial_fitness
                    personal_best[i], personal_best_fitness[i] = trial, trial_fitness  # Update personal best

            # Adaptive learning to adjust mutation factor and crossover rate
            self.crossover_rate = 0.6 + 0.4 * np.random.rand()
        
        best_index = np.argmin(fitness)
        return population[best_index]