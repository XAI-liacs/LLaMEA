import numpy as np

class DualSwarmDynamicMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles_main = 20
        self.num_particles_aux = 15
        self.inertia_weight = 0.8
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.memory_factor = 0.05

    def __call__(self, func):
        num_evaluations = 0
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub

        # Initialize main and auxiliary swarms
        main_positions = np.random.uniform(lb, ub, (self.num_particles_main, self.dim))
        main_velocities = np.random.uniform(-1, 1, (self.num_particles_main, self.dim))
        aux_positions = np.random.uniform(lb, ub, (self.num_particles_aux, self.dim))
        personal_best_positions = np.copy(main_positions)
        personal_best_scores = np.array([float('inf')] * self.num_particles_main)

        # Evaluate initial solutions for main swarm
        for i in range(self.num_particles_main):
            score = func(main_positions[i])
            num_evaluations += 1
            personal_best_scores[i] = score
            if num_evaluations >= self.budget:
                return main_positions[i]

        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        # Main loop
        while num_evaluations < self.budget:
            for swarm, positions, velocities, num_particles in (
                ('main', main_positions, main_velocities, self.num_particles_main), 
                ('aux', aux_positions, np.zeros_like(aux_positions), self.num_particles_aux)
            ):
                for i in range(num_particles):
                    # Update velocity and position
                    if swarm == 'main':
                        r1, r2 = np.random.rand(), np.random.rand()
                        velocities[i] = (self.inertia_weight * velocities[i] +
                                         self.cognitive_param * r1 * (personal_best_positions[i] - positions[i]) +
                                         self.social_param * r2 * (global_best_position - positions[i]))
                    else:
                        velocities[i] = self.memory_factor * (global_best_position - positions[i])

                    positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                    # Evaluate
                    score = func(positions[i])
                    num_evaluations += 1

                    if swarm == 'main' and score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = positions[i]

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i]

                    if num_evaluations >= self.budget:
                        return global_best_position

        return global_best_position