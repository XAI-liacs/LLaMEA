import numpy as np

config = {
    "archive_prob": 0.2079936761167,
    "archive_size": 17,
    "cauchy_prob": 0.0369235968885,
    "cauchy_scale": 0.0477448721342,
    "cma_decay": 0.8402236627992,
    "cr": 0.8258784307474,
    "cr_adapt": False,
    "f": 0.457700046307,
    "f_adapt": True,
    "initial_step_size": 0.3425415050515,
    "learning_rate": 0.0361814273267,
    "ortho_rate": 0.0348856611909,
    "pop_size": 10,
    "step_size_adapt": False,
}


class AdaptiveDE_Ortho_CMA_Cauchy_StepSize:
    def __init__(
        self,
        budget=10000,
        dim=10,
        pop_size=50,
        archive_size=10,
        cr=0.5,
        f=0.7,
        ortho_rate=0.1,
        archive_prob=0.1,
        learning_rate=0.1,
        cma_decay=0.9,
        cauchy_prob=0.1,
        cauchy_scale=0.1,
        cr_adapt=True,
        f_adapt=True,
        step_size_adapt=True,
        initial_step_size=0.1,
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.cr = cr
        self.f = f
        self.ortho_rate = ortho_rate
        self.archive_prob = archive_prob
        self.learning_rate = learning_rate
        self.cma_decay = cma_decay
        self.cauchy_prob = cauchy_prob
        self.cauchy_scale = cauchy_scale
        self.cr_adapt = cr_adapt
        self.f_adapt = f_adapt
        self.step_size_adapt = step_size_adapt
        self.initial_step_size = initial_step_size
        self.step_size = initial_step_size  # Initialize step size
        self.population = None
        self.fitness = None
        self.archive = []
        self.mean = None
        self.C = None
        self.cr_history = []
        self.f_history = []
        self.success_history = []  # Track successful steps for step size adaptation

    def __orthogonal_design(self, n):
        if n == 2:
            return np.array([[1, 1], [1, -1]])
        elif n == 4:
            return np.array(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
            )
        elif n == 8:
            hadamard_2 = self.__orthogonal_design(2)
            hadamard_4 = self.__orthogonal_design(4)
            return np.kron(hadamard_4, hadamard_2)
        return None

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim)
        )
        self.fitness = np.array([func(x) for x in self.population])
        self.archive = []
        self.mean = np.mean(self.population, axis=0)
        self.C = np.eye(self.dim)
        evals = self.pop_size

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                # Mutation: Combined Cauchy and Gaussian
                if np.random.rand() < self.cauchy_prob:
                    # Cauchy mutation
                    mutant = self.population[
                        i
                    ] + self.cauchy_scale * np.random.standard_cauchy(size=self.dim)
                else:
                    # Gaussian mutation
                    mutant = self.population[
                        i
                    ] + self.f * np.random.multivariate_normal(
                        np.zeros(self.dim), self.C
                    )
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Archive interaction
                if len(self.archive) > 0 and np.random.rand() < self.archive_prob:
                    arc_idx = np.random.randint(len(self.archive))
                    mutant = self.archive[arc_idx]

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, self.population[i])

                # Orthogonal Crossover based on promising region
                if np.random.rand() < self.ortho_rate:
                    design_size = 2
                    if self.dim >= 4:
                        design_size = 4
                    if self.dim >= 8:
                        design_size = 8

                    hadamard_matrix = self.__orthogonal_design(design_size)

                    if hadamard_matrix is not None:
                        best_index = np.argmin(self.fitness)
                        x_best = self.population[best_index]

                        num_factors = min(design_size, self.dim)
                        selected_indices = np.random.choice(
                            self.dim, num_factors, replace=False
                        )
                        levels = np.array([-1, 1])  # Define levels as -1 and 1
                        factor_levels = np.zeros((design_size, self.dim))

                        for j in range(num_factors):
                            factor_levels[:, selected_indices[j]] = hadamard_matrix[
                                :, j
                            ]

                        trial_fitnesses = []
                        trial_vectors = []
                        for row in factor_levels:
                            orthogonal_vector = self.population[i].copy()
                            for k in range(self.dim):
                                if row[k] != 0:
                                    orthogonal_vector[k] = x_best[k] + row[
                                        k
                                    ] * self.step_size * (
                                        self.population[i][k] - x_best[k]
                                    )  # Use step_size
                            orthogonal_vector = np.clip(
                                orthogonal_vector, func.bounds.lb, func.bounds.ub
                            )
                            trial_vectors.append(orthogonal_vector)
                            trial_fitnesses.append(func(orthogonal_vector))
                            evals += 1
                            if evals >= self.budget:
                                break

                        best_index = np.argmin(trial_fitnesses)

                        if trial_fitnesses[best_index] < self.fitness[i]:
                            trial = trial_vectors[best_index].copy()

                # Evaluation
                f_trial = func(trial)
                evals += 1

                if f_trial < self.fitness[i]:
                    # Archive update
                    if len(self.archive) < self.archive_size:
                        self.archive.append(self.population[i].copy())
                    else:
                        idx_to_replace = np.random.randint(len(self.archive))
                        self.archive[idx_to_replace] = self.population[i].copy()

                    # Step size adaptation based on success rate
                    self.success_history.append(1)  # Record success
                    if self.step_size_adapt:
                        # Adjust step size based on recent successes.
                        success_rate = np.mean(
                            self.success_history[-min(len(self.success_history), 10) :]
                        )  # Average over last 10 successes
                        if success_rate > 0.6:
                            self.step_size *= 1.1  # Increase step size if successful
                        elif success_rate < 0.2:
                            self.step_size *= (
                                0.9  # Decrease step size if not successful
                            )
                        self.step_size = np.clip(
                            self.step_size, 0.01, 1.0
                        )  # Clip the step size to a reasonable interval

                    # Adaptation of parameters
                    if self.cr_adapt:
                        self.cr_history.append(self.cr)
                        self.cr = np.random.normal(0.5, 0.1)
                        self.cr = np.clip(self.cr, 0.1, 0.9)
                    if self.f_adapt:
                        self.f_history.append(self.f)
                        self.f = np.random.normal(0.7, 0.1)
                        self.f = np.clip(self.f, 0.1, 0.9)

                    # Population update
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    # Simplified CMA update with rank-one update
                    diff = self.population[i] - self.mean
                    self.C = (self.cma_decay) * self.C + (
                        1 - self.cma_decay
                    ) * np.outer(diff, diff)
                    self.mean = np.mean(self.population, axis=0)

                    # Update best solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    self.success_history.append(0)  # Record failure

        return self.f_opt, self.x_opt


# Space:
