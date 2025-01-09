import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.c1 = 2.05
        self.c2 = 2.05
        self.w_max = 0.9  # Adapted max inertia weight
        self.w_min = 0.4  # Adapted min inertia weight
        self.F = 0.8
        self.CR = 0.9
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        
        # Initialize particles
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        # Evaluate initial particle positions
        for i in range(self.swarm_size):
            score = func(positions[i])
            personal_best_scores[i] = score

        # Identify the global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]

        evaluations = self.swarm_size
        best_score = np.min(personal_best_scores)

        while evaluations < self.budget:
            # Update inertia weight dynamically
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            
            for i in range(self.swarm_size):
                # Update velocities and positions using PSO rules
                r1, r2 = np.random.rand(2)
                velocities[i] = w * velocities[i] + \
                                self.c1 * r1 * (personal_best_positions[i] - positions[i]) + \
                                self.c2 * r2 * (global_best_position - positions[i])
                positions[i] = positions[i] + velocities[i]
                
                # Apply boundary conditions
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1
                
                if evaluations >= self.budget:
                    break
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < best_score:
                    global_best_position = positions[i]
                    best_score = score

            # Hybrid with Differential Evolution mutation strategy
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                
                idxs = [idx for idx in range(self.swarm_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = personal_best_positions[a] + self.F * (personal_best_positions[b] - personal_best_positions[c])
                trial_vector = np.copy(positions[i])

                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial_vector[j] = mutant_vector[j]

                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])

                # Evaluate the trial vector
                score = func(trial_vector)
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = trial_vector
                
                # Update global best
                if score < best_score:
                    global_best_position = trial_vector
                    best_score = score

        return global_best_position