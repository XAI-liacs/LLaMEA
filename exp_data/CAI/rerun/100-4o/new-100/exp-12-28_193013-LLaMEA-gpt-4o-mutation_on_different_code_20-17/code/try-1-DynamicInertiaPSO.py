import numpy as np

class DynamicInertiaPSO:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1.5
        self.c2_max = 2.5
        self.c2_min = 1.5
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])

        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        eval_count = self.num_particles

        while eval_count < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (eval_count / self.budget)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (eval_count / self.budget)

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 c2 * r2 * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

                f_value = func(particles[i])
                eval_count += 1

                if f_value < personal_best_values[i]:
                    personal_best_values[i] = f_value
                    personal_best_positions[i] = particles[i]

                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_position = particles[i]

                if eval_count % (self.budget // 10) == 0:  # Random restart mechanism
                    particles[i] = np.random.uniform(lb, ub, self.dim)
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
                
                if eval_count >= self.budget:
                    break
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt