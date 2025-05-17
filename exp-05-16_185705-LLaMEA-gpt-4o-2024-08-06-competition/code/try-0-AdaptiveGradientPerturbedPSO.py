import numpy as np

class AdaptiveGradientPerturbedPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, c1=1.5, c2=1.5, w=0.7):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def __call__(self, func):
        np.random.seed(42)
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize particles
        particles = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Calculate gradient approximation
                epsilon = 1e-8
                approx_gradient = np.zeros(self.dim)
                perturbation = np.random.normal(0, 1, self.dim)
                for d in range(self.dim):
                    perturbed = np.copy(particles[i])
                    perturbed[d] += epsilon * perturbation[d]
                    approx_gradient[d] = (func(perturbed) - personal_best_scores[i]) / (epsilon * perturbation[d])
                    evaluations += 1
                    if evaluations >= self.budget:
                        break

                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]) +
                                 0.1 * approx_gradient)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], -100, 100)

                # Evaluate new position
                score = func(particles[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = score

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt