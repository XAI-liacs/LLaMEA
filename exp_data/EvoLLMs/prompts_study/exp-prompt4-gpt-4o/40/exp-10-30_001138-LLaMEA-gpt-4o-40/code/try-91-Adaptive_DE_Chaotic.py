import numpy as np

class Adaptive_DE_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.8  # Adapted for better exploration-exploitation balance
        self.inertia_damping = 0.9
        self.cognitive_coeff = 1.2  # Adjusted for improved local search
        self.social_coeff = 1.4  # Optimized for dynamic exploration
        self.mutation_factor = 0.8  # Balanced for controlled diversity
        self.crossover_rate = 0.7  # Tuned for robust exploitation
        self.elite_fraction = 0.15  # Reduced for more exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0
        self.initial_pop_size = self.pop_size
        self.chaotic_sequence = self.init_chaotic_sequence()

    def init_chaotic_sequence(self):
        x = np.random.rand()
        sequence = [x]
        for _ in range(self.budget):  # Reduced sequence length for efficiency
            x = 4 * x * (1 - x)
            sequence.append(x)
        return np.array(sequence)

    def __call__(self, func):
        chaotic_idx = 0
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.population)
            self.evaluations += self.pop_size

            better_scores_idx = scores < self.personal_best_scores
            self.personal_best_positions[better_scores_idx] = self.population[better_scores_idx]
            self.personal_best_scores[better_scores_idx] = scores[better_scores_idx]

            min_idx = np.argmin(scores)
            if scores[min_idx] < self.global_best_score:
                self.global_best_score = scores[min_idx]
                self.global_best_position = self.population[min_idx]

            self.inertia_weight *= self.inertia_damping
            r1, r2 = self.chaotic_sequence[chaotic_idx:chaotic_idx+self.pop_size], self.chaotic_sequence[chaotic_idx+self.pop_size:chaotic_idx+2*self.pop_size]
            chaotic_idx = (chaotic_idx + self.pop_size * 2) % len(self.chaotic_sequence)  # Adaptive chaos control
            cognitive_component = self.cognitive_coeff * r1[:, np.newaxis] * (self.personal_best_positions - self.population)
            social_component = self.social_coeff * r2[:, np.newaxis] * (self.global_best_position - self.population)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            elite_count = int(self.elite_fraction * self.pop_size)
            elite_indices = np.argsort(scores)[:elite_count]
            for i in range(self.pop_size):
                if i in elite_indices:
                    continue
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < scores[i]:
                    self.population[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_positions[i] = trial_vector
                        self.personal_best_scores[i] = trial_score
                        if trial_score < self.global_best_score:
                            self.global_best_score = trial_score
                            self.global_best_position = trial_vector
            
            if self.evaluations > self.budget * 0.6:
                self.pop_size = max(20, int(self.initial_pop_size * (self.budget - self.evaluations) / self.budget))
                self.population = self.population[:self.pop_size]
                self.velocities = self.velocities[:self.pop_size]
                self.personal_best_positions = self.personal_best_positions[:self.pop_size]
                self.personal_best_scores = self.personal_best_scores[:self.pop_size]

            if np.random.rand() < 0.05:  # Reduced frequency for Levy flights
                step = self.levy_flight((self.pop_size, self.dim))
                self.population += step
                self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            if self.evaluations % (self.budget // 10) == 0:  # Less frequent reinitialization
                stagnant_indices = scores > np.median(scores)
                self.population[stagnant_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (stagnant_indices.sum(), self.dim))

        return self.global_best_position, self.global_best_score