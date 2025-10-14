import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.stats import special_ortho_group
from scipy.optimize import root_scalar
import torch
import functools
import jax
import jax.numpy as jnp


def good_plt_config():
    plt.style.use("default")
    with open("latex-preambula.tex", "r") as f:
        latex_preambula = f.read()
    plt.rcParams["text.usetex"] = True
    plt.rc("text.latex", preamble=latex_preambula)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = (0, (5, 5))
    plt.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["font.size"] = 20
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


def default_plt_config():
    plt.style.use("default")


def compose_points(f, x, y):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i][j] = f(np.array([X[i][j], Y[i][j]]))
    return X, Y, Z


def compute_zax_min_max(zax_min, zax_max, zfactor=1.0, inversion=False):
    h = zax_max - zax_min
    shift = 2 * h * zfactor
    if inversion:
        return zax_max + shift, zax_min
    return zax_min - shift, zax_max


def rastrigin(x):
    N = len(x)
    # c = 2*math.pi
    c = 1.5
    return 10 * N + sum(x[i] ** 2 - 10 * np.cos(c * x[i]) for i in range(N))


def sphere(x):
    return sum((xi - 1) ** 2 for xi in x)


def truncated_exponential_mean(lambda_, b):
    numerator = 1 - np.exp(-lambda_ * b) * (1 + lambda_ * b)
    denominator = lambda_ * (1 - np.exp(-lambda_ * b))
    return numerator / denominator


def find_lambda_for_mean(mean, b):
    def objective(lambda_):
        return truncated_exponential_mean(lambda_, b) - mean

    sol = root_scalar(objective, bracket=[1e-6, 1e6], method="brentq")
    return sol.root


class GallagherFunction:
    def __init__(
        self,
        num_peaks: int = 10,
        dim: int = 2,
        min_peak_distance: float = 2.0,
        seed: int | None = None,
        max_weight: float = 100.0,
        min_cond_number: float = 1.0,
        mean_cond_number: float = 5.0,
        max_cond_number: float = 1e3,
        min_dist: float = 1.0,
        max_dist: float = 2.0,
    ):
        """
        Initialize parameters for the Gallagher multimodal test function.

        Parameters
        ----------
        num_peaks : int, optional
            Number of local optima (“peaks”) in the search space. Default is 10.
        dim : int, optional
            Dimensionality of the search space (number of variables). Default is 2.
        min_peak_distance : float, optional
            Minimum Euclidean distance between any two peaks to ensure they
            are sufficiently separated. Default is 2.0.
        seed : int or None, optional
            Seed for the random number generator to ensure reproducibility.
            If None, a random seed is used. Default is None.
        max_weight : float, optional
            Maximum “height” of a peak -- i.e., the absolute objective value of
            the smallest local minimum. Controls the contrast between optima.
            Default is 100.0.
        min_cond_number : float, optional
            Minimum condition number of the ellipsoidal attraction basins
            around the optima, determining how “stretched” each basin is.
            Default is 1.0.
        mean_cond_number : float, optional
            Mean condition number of all basins, used when sampling the
            distribution of condition numbers. Default is 5.0.
        max_cond_number : float, optional
            Maximum condition number for any basin. Default is 1e3.
        min_dist : float, optional
            Minimum radius from a basin's center at which the function value
            reaches -1, controlling basin width. Default is 1.0.
        max_dist : float, optional
            Maximum such radius, setting an upper bound on basin width.
            Default is 2.0.

        Notes
        -----
        - Varying the condition numbers allows modeling basins from nearly
          spherical to highly elongated shapes.
        - The min_dist and max_dist parameters guarantee the function
          attains the value -1 within the specified radius of each peak.
        """
        self.num_peaks = num_peaks
        self.dim = dim
        self.min_peak_distance = min_peak_distance
        self.rng = np.random.default_rng(seed)
        self.max_weight = max_weight
        self.min_cond_number = min_cond_number
        self.mean_cond_number = mean_cond_number
        self.max_cond_number = max_cond_number
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.__build_function()
        self.__build_for_jax()

    def __build_function(self):
        # generate peaks without overlap
        self.peaks = []
        while len(self.peaks) < self.num_peaks:
            candidate = self.rng.uniform(-4.5, 4.5, size=self.dim)
            if all(
                np.linalg.norm(candidate - p) >= self.min_peak_distance
                for p in self.peaks
            ):
                self.peaks.append(candidate)
        self.peaks = np.array(self.peaks)

        lam = find_lambda_for_mean(self.mean_cond_number, self.max_cond_number)
        sample_trunc_exp = (
            lambda lam: self.min_cond_number
            - np.log(
                1
                - self.rng.uniform(size=1)[0]
                * (1 - np.exp(-lam * self.max_cond_number))
            )
            / lam
        )

        self.weight = np.ones(self.num_peaks) * self.max_weight
        self.covs = []
        self.covs_t = []
        self.cond_numbers = []
        self.max_eigvalues = []
        self.min_eigvalues = []

        for i in range(self.num_peaks):
            cond_number = sample_trunc_exp(lam)
            self.cond_numbers.append(cond_number)
            dist = self.rng.uniform(self.min_dist, self.max_dist)
            max_eigv = -2 * np.log(1 / self.weight[i]) / dist
            min_eigv = max_eigv / cond_number
            self.max_eigvalues.append(max_eigv)
            self.min_eigvalues.append(min_eigv)
            eigvals = np.concatenate(
                (
                    np.array([max_eigv, min_eigv]),
                    self.rng.uniform(min_eigv, max_eigv, size=self.dim - 2),
                )
            )
            R = special_ortho_group.rvs(self.dim, random_state=self.rng)
            self.covs.append(R.T @ np.diag(eigvals) @ R)
            cov_t = torch.tensor(self.covs[i], dtype=torch.float64)
            self.covs_t.append(cov_t)

    def __build_for_jax(self):
        self.peaks_jax = jnp.stack(self.peaks)
        self.covs_jax = jnp.stack(self.covs)
        self.weights_jax = jnp.asarray(self.weight)

    def __call__(self, x: np.ndarray):
        ans = 0
        for i, x_star in enumerate(self.peaks):
            qf = np.exp(-0.5 * (x - x_star).T @ self.covs[i] @ (x - x_star))
            ans = max(
                ans,
                self.weight[i] * qf,
            )
        return -ans

    @functools.partial(jax.jit, static_argnums=0)
    def call_jax(self, x):
        diffs = x[None, :] - self.peaks_jax

        # einsum to get [diffs[i] @ covs[i] @ diffs[i]] for each i
        # ‘nd,ndd,nd->n’ means: for each n, contract diffs[n] (d) with covs[n] (d,d) and diffs[n] (d)
        quad_terms = jnp.einsum("nd,ndd,nd->n", diffs, self.covs_jax, diffs)

        qfs = jnp.exp(-0.5 * quad_terms)

        # weighted values and take the negative max
        vals = self.weights_jax * qfs
        return -jnp.max(vals)

    @functools.partial(jax.jit, static_argnums=0)
    def make_gradient_jax(self, x):
        # grad wants a function whose first arg is x,
        # but our method has signature (self, x), so:
        df_dx = jax.grad(lambda x: self.call_jax(x))
        return df_dx

    def _torch_eval(self, x_t: torch.Tensor) -> torch.Tensor:
        vals = []
        for i, x_star in enumerate(self.peaks):
            x_star_t = torch.tensor(x_star, dtype=torch.float64)
            diff = x_t - x_star_t
            qf = torch.exp(-0.5 * diff @ self.covs_t[i] @ diff)
            vals.append(self.weight[i] * qf)
        return -torch.max(torch.stack(vals))

    def gradient_torch(self, x) -> np.ndarray:
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        ans = self._torch_eval(x_t)
        ans.backward()
        return x_t.grad.numpy()

    def hessian_torch(self, x) -> np.ndarray:
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        scalar_val = lambda inp: self._torch_eval(inp)
        H = torch.autograd.functional.hessian(scalar_val, x_t)
        return H.numpy()


def plot_3D_surface(
    x1_lims,
    x2_lims,
    f,
    X_data,
    discretization=50,
    is_colorbar=False,
    is_axis_names=False,
    is_white_facecolor=False,
    is_scatter_search_space=True,
    is_scatter_objective_space=False,
    zfactor=1.0,
    is_inverse=False,
    is_remove_extra_zlables=False,
    zlabelpad=10,
    is_connect_search_obj=False,
):
    x1 = np.linspace(*x1_lims, discretization)
    x2 = np.linspace(*x2_lims, discretization)
    X1, X2, Z = compose_points(f, x1, x2)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
    fig.set_size_inches(18.5, 10.5)
    mycmap = mpl.cm.jet
    ax.plot_surface(
        X1,
        X2,
        Z,
        cmap=mycmap,
        antialiased=True,
        linewidth=0.2,
        edgecolor="k",
        rcount=len(X1),
        ccount=len(X1[0]),
        alpha=1,
        zorder=3,
    )
    zax_min, zax_max = ax.get_zlim()
    zax_min, zax_max = compute_zax_min_max(zax_min, zax_max, zfactor, is_inverse)
    ax.set_zlim(zax_min, zax_max)

    if is_remove_extra_zlables:
        ticks = ax.get_zticks()
        labels = ax.get_zticklabels()
        if is_inverse:
            is_prv = False
            _zmax = Z.max()
            for tick_val, label in zip(ticks, labels):
                if is_prv:
                    label.set_visible(False)
                if tick_val > _zmax:
                    is_prv = True
        else:
            _zmin = 1.1 * Z.min()
            for tick_val, label in zip(ticks, labels):
                if tick_val < _zmin:
                    label.set_visible(False)

    ax.contourf(
        X1,
        X2,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mycmap,
        extend="both",
        levels=50,
        alpha=0.4,
        zorder=2,
    )
    ax.contour(
        X1,
        X2,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mycmap,
        # colors='k',
        extend="both",
        levels=50,
        alpha=1,
        zorder=2,
        linewidths=0.5,
        # linestyles='--',
    )
    if len(X_data) > 0:
        y_data = [f(x) for x in X_data]
        if is_scatter_objective_space:
            ax.scatter(
                X_data[:, 0],
                X_data[:, 1],
                y_data,
                c="magenta",
                marker="+",
                s=100,
                alpha=1,
                zorder=4,
            )
        if is_scatter_search_space:
            ax.scatter(
                X_data[:, 0],
                X_data[:, 1],
                zax_min,
                c="red",
                marker="+",
                s=100,
                alpha=1,
                zorder=4,
            )
        if (
            is_scatter_search_space
            and is_scatter_search_space
            and is_connect_search_obj
        ):
            for i, x in enumerate(X_data):
                ax.plot(
                    [x[0], x[0]],  # X stays constant
                    [x[1], x[1]],  # Y stays constant
                    [zax_min, y_data[i]],  # Z goes from z1 to z2
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    zorder=2,
                )
    if is_colorbar:
        zmin, zmax = Z.min(), Z.max()
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(zmin, zmax))
        fig.colorbar(
            sm,
            ax=ax,
            shrink=0.7,
            ticks=np.linspace(zmin, zmax, 5),
            orientation="vertical",
            extend="both",
        )
    if is_axis_names:
        ax.set_xlabel(r"$x_1$", labelpad=10)
        ax.set_ylabel(r"$x_2$", labelpad=10)
        zlabel = r"$f\!\br{\bm{x}}$"
        if is_inverse:
            zlabel = r"$\text{\textbf{\textcolor{red}{(inversed)}}}$ " + zlabel
        ax.set_zlabel(zlabel, labelpad=zlabelpad)
    if is_white_facecolor:
        ax.get_xaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.get_yaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.get_zaxis().set_pane_color((1.0, 1.0, 1.0, 1.0))
    return fig, ax


def parse_points(file_name):
    _X = np.genfromtxt(file_name, delimiter=",")
    if _X.ndim == 1:
        _X = np.array([_X])
    X = _X[:, 1:]
    return [x for x in X]


class BasinsVis:

    def __init__(self, minr):
        self.id_to_point = []
        self.EPS_INIT = 1e-10
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.EPS_FINAL = minr

    @staticmethod
    def d(p1, p2):
        return np.linalg.norm(p1 - p2)

    def point_to_id(self, p):
        for i in range(len(self.id_to_point)):
            if BasinsVis.d(self.id_to_point[i], p) < self.EPS_INIT:
                return i
        return None

    def parse_path_from_line(self, s):
        path_points = []
        pos = 0
        assert s[pos] == "{"
        pos += 1
        while pos < len(s):
            if s[pos] == "[":
                pos += 1
                parsed_list = []
                v = ""
                while True:
                    if s[pos] == "," or s[pos] == "]":
                        parsed_list.append(float(v))
                        v = ""
                        if s[pos] == "]":
                            break
                    else:
                        v += s[pos]
                    pos += 1
                assert s[pos] == "]"
                path_points.append(np.array(parsed_list))
            pos += 1
        return path_points

    def parse_path_points(self, file_name):
        path_points = []
        with open(self.root + "/" + file_name, "r") as f:
            for s in f:
                path_points += self.parse_path_from_line(s)
        return path_points

    def parse_paths_to_graph(self, file_name):
        g = [[] for _ in range(len(self.id_to_point))]
        with open(self.root + "/" + file_name, "r") as f:
            for s in f:
                path = self.parse_path_from_line(s)
                prv_id_p = None
                for p in path:
                    id_p = self.point_to_id(p)
                    if prv_id_p and id_p not in g[prv_id_p]:
                        g[prv_id_p].append(id_p)
                    prv_id_p = id_p
        return g

    def give_ids(self, ps):
        my_id_to_point = []
        for p in ps:
            has_id = False
            for p1 in my_id_to_point:
                if BasinsVis.d(p, p1) < self.EPS_INIT:
                    has_id = True
                    break
            if not has_id:
                my_id_to_point.append(p)
        return my_id_to_point

    def transpose(self, G):
        G_T = [[] for _ in range(len(G))]
        for u in range(len(G)):
            for v in G[u]:
                G_T[v].append(u)
        return G_T

    def dfs(self, G, v, leaves):
        for u in G[v]:
            if u != v:
                self.dfs(G, u, leaves)
        if not len(G[v]):
            leaves.append(v)

    def dfs1(self, g, v, ids_init_points, ids_final_points, init_nodes):
        for u in g[v]:
            if u != v:
                self.dfs1(g, u, ids_init_points, ids_final_points, init_nodes)
        if v in ids_init_points and v not in ids_final_points:
            init_nodes.append(v)

    def find_leaves(self, G, v):
        leaves = []
        self.dfs(G, v, leaves)
        return leaves

    def find_init_nodes(self, g, v, ids_init_points, ids_final_points):
        init_nodes = []
        self.dfs1(g, v, ids_init_points, ids_final_points, init_nodes)
        return init_nodes

    def process_paths(self):
        self.points_initial = parse_points("points_c.csv")  # init points
        self.points_final = parse_points("points_c_1.csv")  # final points
        self.points_paths = self.parse_path_points("paths.txt")
        self.all_points = self.points_initial + self.points_final + self.points_paths
        self.id_to_point = self.give_ids(self.all_points)
        self.ids_points_initial = [self.point_to_id(p) for p in self.points_initial]
        self.ids_points_final = [self.point_to_id(p) for p in self.points_final]

        self.G = self.parse_paths_to_graph("paths.txt")
        # G_T = transpose(G)

        mp = {}
        self.mp_id = np.zeros(len(self.id_to_point), dtype=int)
        for i in range(len(self.mp_id)):
            self.mp_id[i] = i
        for id_p_final_1 in self.ids_points_final:
            all_similar = []
            me = self.id_to_point[id_p_final_1]
            for another_id in self.id_to_point:
                if BasinsVis.d(self.id_to_point[another_id], me) < self.EPS_FINAL:
                    all_similar.append(another_id)
            for id_p_final in all_similar:
                init_ids = self.find_init_nodes(
                    self.G, id_p_final, self.ids_points_initial, self.ids_points_final
                )
                for init_id in init_ids:
                    if init_id not in mp:
                        mp[init_id] = [
                            self.id_to_point[init_id],
                            self.id_to_point[id_p_final],
                        ]
                        self.mp_id[init_id] = id_p_final
                    else:
                        raise RuntimeError(
                            f"The point with the same init_id={init_id} occures twice"
                        )

        with open("init_point_to_final_point.txt", "w") as f:
            for ps in mp.values():
                print(ps[0].tolist(), ps[1].tolist(), file=f)
        return mp

    def chart_with_arrows(self):
        cmap = plt.get_cmap("jet")

        id_point_final_to_id_color = np.zeros(max(self.ids_points_final) + 1, dtype=int)
        cnt = 0
        for final_id in self.ids_points_final:
            id_point_final_to_id_color[final_id] = cnt
            cnt += 1
        id_point_init_to_id_color = np.zeros(
            max(self.ids_points_initial) + 1, dtype=int
        )
        for init_id in self.ids_points_initial:
            id_point_init_to_id_color[init_id] = id_point_final_to_id_color[
                self.mp_id[init_id]
            ]

        X_init = []
        color_init = []
        X_final = []
        color_final = []

        def get_color(id_final_color):
            denom = cnt - 1 if cnt - 1 != 0 else 1
            return cmap(id_final_color / denom)

        cnt = 0
        for id_init in self.ids_points_initial:
            X_init.append(self.id_to_point[id_init])
            color_init.append(get_color(id_point_init_to_id_color[id_init]))
        for id_final in self.ids_points_final:
            X_final.append(self.id_to_point[id_final])
            color_final.append(get_color(id_point_final_to_id_color[id_final]))

        X_init = np.array(X_init)
        X_final = np.array(X_final)

        for c1, c2 in [(0, 1)]:
            fig, ax = plt.subplots()
            ax.scatter(X_init[:, c1], X_init[:, c2], c=color_init, alpha=1, s=10)
            ax.scatter(
                X_final[:, c1],
                X_final[:, c2],
                marker="*",
                c=color_final,
                alpha=0.3,
                s=200,
            )
            for id_init in self.ids_points_initial:
                id_final = self.mp_id[id_init]
                start_point = (
                    self.id_to_point[id_init][c1],
                    self.id_to_point[id_init][c2],
                )
                end_point = (
                    self.id_to_point[id_final][c1],
                    self.id_to_point[id_final][c2],
                )
                color = get_color(id_point_final_to_id_color[id_final])
                ax.annotate(
                    "",
                    xytext=start_point,
                    xy=end_point,
                    arrowprops=dict(
                        facecolor=color,
                        edgecolor=color,
                        arrowstyle="->",
                    ),
                )
            ax.set_xlabel("$x" + f"_{c1+1}$")
            ax.set_ylabel("$x" + f"_{c2+1}$")
            fig.savefig(f"colored_points-projection-{c1}-{c2}.pdf")
            plt.show()
            plt.close()


def add_arrows_to_xy_in_3d(ax, ids_points_initial, id_to_point, mp_id):
    zax_min, _ = ax.get_zlim()
    for id_init in ids_points_initial:
        id_final = mp_id[id_init]
        start_point = (
            id_to_point[id_init][0],
            id_to_point[id_init][1],
        )
        end_point = (id_to_point[id_final][0], id_to_point[id_final][1])
        ax.scatter(
            end_point[0],
            end_point[1],
            zax_min,
            marker="*",
            c="red",
            alpha=1,
            s=200,
        )
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [zax_min, zax_min],
            color="red",
            zdir="z",
            zorder=4,
        )
