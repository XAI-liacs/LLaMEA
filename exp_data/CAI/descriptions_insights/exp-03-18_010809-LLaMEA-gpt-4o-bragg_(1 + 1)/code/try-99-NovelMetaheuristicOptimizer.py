import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize particles
        particles = np.random.uniform(low=lb, high=ub, size=(self.budget, self.dim))
        velocities = np.random.uniform(low=-1, high=1, size=(self.budget, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.budget, np.inf)

        # Initialize global best
        global_best_score = np.inf
        global_best_position = None

        # Iteration counter
        iterations = 0

        while iterations < self.budget:
            for i in range(self.budget):
                # Evaluate the current position
                score = func(particles[i])

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            # Update velocities and positions
            inertia_weight = 0.5 + (0.5 / (1 + iterations))  # adaptive inertia weight
            cognitive_component = np.random.rand(self.budget, self.dim)
            social_component = np.random.rand(self.budget, self.dim)

            velocities = (
                inertia_weight * velocities +
                cognitive_component * (personal_best_positions - particles) + 
                social_component * (global_best_position - particles)
            )

            # Position update with time-varying random perturbations
            particles += velocities + (0.1 / (1 + iterations)) * np.random.normal(size=(self.budget, self.dim))

            # Ensure particles are within bounds
            particles = np.clip(particles, lb, ub)

            iterations += 1

        return global_best_position, global_best_score