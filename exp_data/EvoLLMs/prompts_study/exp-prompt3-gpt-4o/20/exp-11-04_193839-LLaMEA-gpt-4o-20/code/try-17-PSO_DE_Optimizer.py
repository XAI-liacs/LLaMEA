import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.mutation_factor_min = 0.5
        self.mutation_factor_max = 0.9
        self.crossover_prob = 0.9
        
    def __call__(self, func):
        num_evaluations = 0
        
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        de_population = np.copy(positions)
        
        while num_evaluations < self.budget:
            inertia_weight = (self.final_inertia_weight + 
                              (self.initial_inertia_weight - self.final_inertia_weight) *
                              (self.budget - num_evaluations) / self.budget)
            mutation_factor = (self.mutation_factor_max - self.mutation_factor_min) * np.random.rand() + self.mutation_factor_min
            
            for i in range(self.population_size):
                score = func(positions[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]
            
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                          self.social_coeff * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)
            
            for i in range(self.population_size):
                idx = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant_vector = de_population[idx[0]] + mutation_factor * (de_population[idx[1]] - de_population[idx[2]])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(de_population[i])
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover] = mutant_vector[crossover]
                
                trial_score = func(trial_vector)
                num_evaluations += 1
                if trial_score < func(de_population[i]):
                    de_population[i] = trial_vector
                
                if num_evaluations >= self.budget:
                    break

        return global_best_position, global_best_score