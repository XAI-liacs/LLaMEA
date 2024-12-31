import numpy as np

class HybridPSOSA:
    def __init__(self, budget=10000, dim=10, population_size=50, inertia=0.7, cognitive=1.5, social=1.5, temp=1.0):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.temp = temp
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        # Initialize particle positions and velocities
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(x) for x in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        eval_count = self.population_size
        while eval_count < self.budget:
            for i in range(self.population_size):
                # Update velocities with PSO dynamics
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia * velocities[i]
                                + self.cognitive * r1 * (personal_best_positions[i] - particles[i])
                                + self.social * r2 * (global_best_position - particles[i]))

                # Update positions
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

                # Evaluate objective function
                f = func(particles[i])
                eval_count += 1
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = particles[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = particles[i]
            
            # Simulated annealing component: perturb global best with controlled probability
            perturbation = np.random.uniform(-1, 1, self.dim) * (self.temp / 100)
            perturbed_position = np.clip(global_best_position + perturbation, func.bounds.lb, func.bounds.ub)
            f_perturbed = func(perturbed_position)
            eval_count += 1
            if f_perturbed < self.f_opt or np.exp((self.f_opt - f_perturbed) / self.temp) > np.random.rand():
                global_best_position = perturbed_position
                self.f_opt = f_perturbed
                self.x_opt = perturbed_position
            
            # Cooling schedule for simulated annealing
            self.temp *= 0.99  # cooling schedule
        
        return self.f_opt, self.x_opt