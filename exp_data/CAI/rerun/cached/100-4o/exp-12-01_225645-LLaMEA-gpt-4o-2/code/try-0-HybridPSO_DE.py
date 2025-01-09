import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        # Initialize the swarm
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Particle Swarm Optimization step
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (personal_best_positions - positions) + self.c2 * r2 * (global_best_position - positions)
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # Evaluate new positions
            scores = np.array([func(pos) for pos in positions])
            evaluations += self.population_size
            
            # Update personal bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    
            # Update global best
            current_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_index] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_index]
                global_best_position = personal_best_positions[current_global_best_index]
            
            # Differential Evolution step
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = personal_best_positions[a] + self.F * (personal_best_positions[b] - personal_best_positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(personal_best_positions[i])
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector[crossover] = mutant_vector[crossover]
                
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    
                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial_vector

        return global_best_position, global_best_score