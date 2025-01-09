import numpy as np
import random

class HybridPSO_DE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0  # New: Initial temperature for SA

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # PSO update
            for i in range(self.population_size):
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_weight * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                                 self.social_weight * np.random.rand(self.dim) * (global_best_position - population[i]))
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
            
            # DE update with SA-inspired mutation
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                if random.random() < self.temperature:  # New: SA acceptance probability
                    mutant += np.random.normal(0, 1, self.dim)  # New: Gaussian noise
                trial = np.array([mutant[j] if np.random.rand() < self.crossover_rate else population[i][j] for j in range(self.dim)])
                trial_score = func(trial)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    if trial_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial
            
            if evaluations >= self.budget:
                break
            
            # Update personal and global bests
            for i in range(self.population_size):
                score = func(population[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_scores[i] = score
                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = population[i]

            self.temperature *= 0.99  # New: Cool down temperature

        return global_best_position