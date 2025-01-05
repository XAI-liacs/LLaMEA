import numpy as np
from multiprocessing import Pool

class ParallelQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = max(20, dim * 5)
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.4   # Quantum exploration factor
        self.max_inertia = 0.9
        self.min_inertia = 0.4
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize swarm and velocities
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_scores = np.array([func(p) for p in personal_best_positions])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.swarm_size

        inertia = self.max_inertia

        def update_particle(i):
            nonlocal swarm, velocities, personal_best_positions, personal_best_scores
            r1, r2 = np.random.rand(), np.random.rand()

            # Update velocity
            velocities[i] = (inertia * velocities[i] +
                             self.cognitive_coefficient * r1 * (personal_best_positions[i] - swarm[i]) +
                             self.social_coefficient * r2 * (global_best_position - swarm[i]))

            # Apply quantum-inspired exploration
            if np.random.rand() < self.beta:
                q_step = np.random.normal(0, 1, self.dim) * (ub - lb)
                velocities[i] += q_step

            # Update position
            swarm[i] += velocities[i]
            swarm[i] = np.clip(swarm[i], lb, ub)

            # Evaluate new position
            score = func(swarm[i])
            evaluations += 1

            # Update personal best
            if score < personal_best_scores[i]:
                personal_best_positions[i] = swarm[i]
                personal_best_scores[i] = score

            return score, i

        # Main optimization loop
        while evaluations < self.budget:
            with Pool() as pool:
                results = pool.map(update_particle, range(self.swarm_size))

            # Update global best
            for score, i in results:
                if score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[global_best_index].copy()

            # Adjust inertia weight
            inertia = self.max_inertia - (self.max_inertia - self.min_inertia) * (evaluations / self.budget)

        return global_best_position, personal_best_scores[global_best_index]