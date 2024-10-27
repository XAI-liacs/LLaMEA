import numpy as np

class Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 25
        self.max_iter = 120

    def __call__(self, func):
        def generate_initial_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_particles, self.dim))

        def enhanced_pso_update(particles, velocities, p_best, g_best):
            inertia_weight = 0.5
            c1 = 1.5
            c2 = 1.5

            for i in range(self.num_particles):
                r1, r2 = np.random.random(), np.random.random()
                velocities[i] = inertia_weight * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i] + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)

        def simulated_annealing(current_solution, current_value, best_solution, best_value, temperature, cooling_rate):
            new_solution = current_solution + np.random.normal(0, 0.1, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_value = func(new_solution)

            if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temperature):
                current_solution, current_value = new_solution, new_value

            if new_value < best_value:
                best_solution, best_value = new_solution, new_value

            temperature *= cooling_rate

            return current_solution, current_value, best_solution, best_value, temperature

        particles = generate_initial_population()
        velocities = np.zeros_like(particles)
        p_best = particles.copy()
        g_best = particles[np.argmin([func(p) for p in particles])]
        best_solution = g_best
        best_value = func(g_best)
        temperature = 1.0
        cooling_rate = 0.99

        for _ in range(self.max_iter):
            enhanced_pso_update(particles, velocities, p_best, g_best)
            for i in range(self.num_particles):
                particles[i], _, p_best[i], _, temperature = simulated_annealing(particles[i], func(particles[i]), p_best[i], func(p_best[i]), temperature, cooling_rate)
            
            g_best = particles[np.argmin([func(p) for p in particles])]
            if func(g_best) < best_value:
                best_solution, best_value = g_best, func(g_best)

        return best_solution