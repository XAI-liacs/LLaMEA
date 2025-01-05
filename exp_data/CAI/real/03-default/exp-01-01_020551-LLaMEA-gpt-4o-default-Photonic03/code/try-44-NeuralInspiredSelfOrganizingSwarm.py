import numpy as np

class NeuralInspiredSelfOrganizingSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.learning_rate = 0.1
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.neural_influence = 0.05
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (personal_best_positions[i] - swarm[i])
                social_velocity = self.social_component * r2 * (global_best_position - swarm[i])
                velocities[i] = (self.inertia_weight * velocities[i]
                                + cognitive_velocity
                                + social_velocity)
                
                # Neural-inspired influence
                neural_adjustment = self.neural_influence * np.tanh(velocities[i])
                velocities[i] += neural_adjustment
                
                # Update position
                swarm[i] = np.clip(swarm[i] + velocities[i], lb, ub)
                
                # Evaluation
                score = func(swarm[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]
                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = swarm[i]
                
        return global_best_position, personal_best_scores[global_best_index]