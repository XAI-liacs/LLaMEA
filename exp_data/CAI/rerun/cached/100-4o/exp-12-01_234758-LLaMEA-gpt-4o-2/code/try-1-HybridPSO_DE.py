import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(5 + np.ceil(3 * np.log(dim)))
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize particles and velocities
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)

        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                # PSO update
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] 
                                 + self.c1 * r1 * (personal_best[i] - particles[i])
                                 + self.c2 * r2 * (global_best - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Evaluate particle
                score = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = score
                
                # Update global best
                if score < global_best_score:
                    global_best = particles[i]
                    global_best_score = score

                if evaluations >= self.budget:
                    break

            if evaluations >= self.budget:
                break

            # DE update
            for i in range(self.population_size):
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = personal_best[indices[:3]]
                mutant = x1 + self.f * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(particles[i])
                crossover = np.random.rand(self.dim) < self.cr
                trial[crossover] = mutant[crossover]

                trial_score = func(trial)
                evaluations += 1
                
                # Selection
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial
                    personal_best_scores[i] = trial_score
                
                # Update global best
                if trial_score < global_best_score:
                    global_best = trial
                    global_best_score = trial_score

                if evaluations >= self.budget:
                    break

        return global_best