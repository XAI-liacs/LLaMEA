import numpy as np

class QuantumAdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + int(2 * np.sqrt(dim))
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.q_prob = 0.05  # Probability to perform a quantum-inspired move
        self.inertia_weight = 0.9  # Inertia weight

    def quantum_jump(self, particle, global_best, lb, ub):
        factor = np.random.uniform(-0.1, 0.1, self.dim)
        new_position = global_best + factor * (particle - global_best)
        return np.clip(new_position, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_scores = np.full(self.swarm_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Evaluate current particle
                score = func(swarm[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i].copy()

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i].copy()

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - swarm[i])
                social_component = self.c2 * r2 * (global_best_position - swarm[i])
                
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_component + social_component)
                
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Quantum-inspired jump with a small probability
                if np.random.rand() < self.q_prob:
                    swarm[i] = self.quantum_jump(swarm[i], global_best_position, lb, ub)

        return global_best_position, global_best_score