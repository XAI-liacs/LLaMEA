import numpy as np

class HybridPSO_SA:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.7, cognitive=1.5, social=1.5, temp=1000, cooling_rate=0.99, adapt_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.adapt_radius = adapt_radius
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize the swarm
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.array([func(p) for p in particles])
        
        # Initialize global best
        global_best_index = np.argmin(personal_best_values)
        global_best_position = personal_best_positions[global_best_index]
        global_best_value = personal_best_values[global_best_index]
        
        eval_count = self.swarm_size
        while eval_count < self.budget:
            # Dynamic inertia weight
            dynamic_inertia = 0.5 + (0.5 * (self.budget - eval_count) / self.budget)
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocities = (dynamic_inertia * velocities +
                          self.cognitive * r1 * (personal_best_positions - particles) +
                          self.social * r2 * (global_best_position - particles))
            particles = particles + velocities
            particles = np.clip(particles, lb, ub)
            
            # Evaluate new positions
            for i in range(self.swarm_size):
                f = func(particles[i])
                eval_count += 1
                if f < personal_best_values[i]:
                    personal_best_values[i] = f
                    personal_best_positions[i] = particles[i]
                if f < global_best_value:
                    global_best_value = f
                    global_best_position = particles[i]
            
            # Simulated Annealing refinement
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                # Adaptive neighborhood radius
                new_position = particles[i] + np.random.normal(0, self.adapt_radius, self.dim)
                new_position = np.clip(new_position, lb, ub)
                new_f = func(new_position)
                eval_count += 1
                delta_f = new_f - personal_best_values[i]
                if delta_f < 0 or np.random.rand() < np.exp(-delta_f / self.temp):
                    personal_best_positions[i] = new_position
                    personal_best_values[i] = new_f
                    if new_f < global_best_value:
                        global_best_value = new_f
                        global_best_position = new_position
            
            # Cool down temperature
            self.temp *= self.cooling_rate
        
        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt