import numpy as np

class HybridPSOSA:
    def __init__(self, budget=10000, dim=10, swarm_size=50, inertia=0.5, cognitive=2.0, social=2.0, initial_temp=100.0, cooling_rate=0.99):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        np.random.seed(0)  # For reproducibility
        # Initialize particles
        positions = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        current_temp = self.initial_temp
        evaluations = self.swarm_size  # Initial evaluation count

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i]) +
                                 self.social * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i]))
                
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], -100, 100)

                # Evaluate new positions
                current_score = func(positions[i])
                evaluations += 1

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = positions[i]

                # Simulated Annealing acceptance
                if np.random.rand() < np.exp(-(current_score - personal_best_scores[i]) / current_temp):
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = positions[i]

            # Cooling schedule for Simulated Annealing
            current_temp *= self.cooling_rate

            # Check if budget is exceeded
            if evaluations >= self.budget:
                break

        return global_best_score, global_best_position