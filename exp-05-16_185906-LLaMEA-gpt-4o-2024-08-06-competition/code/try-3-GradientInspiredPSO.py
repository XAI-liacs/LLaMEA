import numpy as np

class GradientInspiredPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive constant
        self.c2 = 1.5  # social constant
        self.alpha = 0.01  # gradient step size

    def __call__(self, func):
        np.random.seed(42)
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize particles
        particles = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])

        # Update global best
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Update velocity and position
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], -100, 100)

                # Evaluate particle and apply gradient descent step
                current_value = func(particles[i])
                if evaluations < self.budget - 1:
                    gradient = self.estimate_gradient(func, particles[i])
                    particles[i] -= self.alpha * gradient
                    particles[i] = np.clip(particles[i], -100, 100)
                    current_value = func(particles[i])
                    evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_values[i] = current_value

                # Update global best
                if current_value < global_best_value:
                    global_best_position = particles[i]
                    global_best_value = current_value

                evaluations += 1
                if evaluations >= self.budget:
                    break

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt

    def estimate_gradient(self, func, x, epsilon=1e-5):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            x_step = np.copy(x)
            x_step[i] += epsilon
            grad[i] = (func(x_step) - func(x)) / epsilon
        return grad