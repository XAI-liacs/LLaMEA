import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, self.budget // 10)
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coefficient * r1 * (personal_best_positions - positions) +
                          self.social_coefficient * r2 * (global_best_position - positions))
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            # Apply a differential evolution step
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = positions[indices]
                mutant_vector = x1 + 0.8 * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                crossover_probability = 0.9
                trial_vector = np.copy(positions[i])
                for d in range(self.dim):
                    if np.random.rand() < crossover_probability or d == np.random.randint(self.dim):
                        trial_vector[d] = mutant_vector[d]
                
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector
        
        return global_best_position