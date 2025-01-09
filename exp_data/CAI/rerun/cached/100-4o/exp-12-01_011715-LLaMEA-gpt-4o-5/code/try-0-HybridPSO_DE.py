import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, 4 + int(3 * np.log(dim)))  # Rule-of-thumb for PSO population size
        self.de_cr = 0.9  # Crossover probability for DE
        self.de_f = 0.8  # Differential weight for DE
        self.c1 = 2.05  # Cognitive coefficient for PSO
        self.c2 = 2.05  # Social coefficient for PSO
        self.w = 0.7  # Inertia weight for PSO

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize particles and velocities for PSO
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)

        # Initialize the global best
        global_best_position = None
        global_best_value = np.inf

        # Main loop of the hybrid algorithm
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate current solutions
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_value = func(particles[i])
                evaluations += 1
                
                # Update personal bests
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]
                
                # Update global best
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles[i]

            # PSO update: Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - particles) +
                          self.c2 * r2 * (global_best_position - particles))
            particles = particles + velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            # DE update: Apply differential evolution strategy
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[indices[0]], particles[indices[1]], particles[indices[2]]
                trial_vector = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.de_cr
                trial_vector = np.where(crossover_mask, trial_vector, particles[i])

                # Evaluate trial vector
                trial_value = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_value < personal_best_values[i]:
                    personal_best_values[i] = trial_value
                    personal_best_positions[i] = trial_vector

                if trial_value < global_best_value:
                    global_best_value = trial_value
                    global_best_position = trial_vector

        return global_best_position, global_best_value