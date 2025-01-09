import numpy as np

class PSO_ADE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive parameter for PSO
        self.c2 = 1.5  # Social parameter for PSO

    def __call__(self, func):
        self._initialize_particles()

        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                # PSO Update
                r1, r2 = np.random.uniform(0, 1, 2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate particle
                score = func(self.particles[i])
                evals += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

                # ADE Update (Mutation and Crossover)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(self.particles[a] + self.F * (self.particles[b] - self.particles[c]), self.lower_bound, self.upper_bound)
                trial = np.copy(self.particles[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Evaluate trial vector
                trial_score = func(trial)
                evals += 1
                if trial_score < score:
                    self.particles[i] = trial
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

                if evals >= self.budget:
                    break

        return self.global_best_position, self.global_best_score

    def _initialize_particles(self):
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        for i in range(self.population_size):
            score = func(self.particles[i])
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i]
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.particles[i]