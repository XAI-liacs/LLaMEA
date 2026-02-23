import numpy as np
import utils


class BasinsLoc:
    def __init__(self, MIN_DIST=2e-3, is_verbose=False):
        self.MIN_DIST = MIN_DIST
        self.verbose = is_verbose

    @staticmethod
    def dist(x: np.ndarray, y: np.ndarray):
        return np.linalg.norm(x - y)

    def opoES(self, f, x, threshold, budget, sigma, lb_sigma, T, gamma):
        dim = len(x)
        value_x = f(x)
        success = 0
        for i in range(1, budget):
            if value_x < threshold:
                return x, value_x
            z = np.random.normal(0, 1, dim)
            y = x + sigma * z
            value_y = f(y)
            if value_y <= value_x:
                success += 1
                x = y
                value_x = value_y
            if i % T == 0:
                if success / T >= 1 / 5:
                    sigma = max(lb_sigma, sigma / gamma)
                else:
                    sigma = sigma * gamma
                success = 0
        return x, value_x

    def find_step(self, p1, v1, p2, v2, f, R, alg="opoES"):
        dp1p2 = BasinsLoc.dist(p1, p2)

        def transform(x):
            if type(x) is not np.ndarray:
                x = np.array(x)
            s = np.sqrt(sum(x**2))
            return p1 + x / s * R

        def myobj(x):
            y = transform(x)
            dyp2 = BasinsLoc.dist(y, p2)
            if dyp2 > dp1p2:
                value = 1 + np.exp(dyp2)
            else:
                value = f(y)
            if self.verbose:
                print(f"dist_y_p2 = {dyp2}, value = {value}")
            return value

        x0 = (p2 - p1) / dp1p2
        x, value = self.opoES(myobj, x0, v1, 100, 0.001, 0.0005, 5, 1.027457)
        return transform(x), value

    class SearchResult:
        def __init__(self):
            self.status = None
            self.x = None
            self.fun = None
            self.path = []

    def dfs(self, point1, value_point1, point2, value_point2, f, ret, MINR):
        curd = BasinsLoc.dist(point1, point2)
        if self.verbose:
            print(f"DFS call: cur dist {curd}, value_point1 = {value_point1}")
        ret.path.append(point1)
        if curd < MINR:
            if self.verbose:
                print(f"Found path from point1 to point2!")
            if value_point1 < value_point2:
                ret.status, ret.x, ret.fun = "PathFound", point1, value_point1
            else:
                ret.status, ret.x, ret.fun = "PathFound", point2, value_point2
                ret.path.append(point2)
            return ret
        if value_point1 < value_point2:
            if self.verbose:
                print(f"Found better point than both! {value_point1}")
            ret.status, ret.x, ret.fun = "ImprovementOfBothFound", point1, value_point1
            return ret
        for i in range(2):
            z, value_z = self.find_step(
                point1, value_point1, point2, value_point2, f, MINR, alg="opoES"
            )
            if value_z < value_point1:
                break
        if value_z < value_point1:
            ret = self.dfs(z, value_z, point2, value_point2, f, ret, MINR)
            if ret.status == "Failed":
                if self.verbose:
                    print(f"Found better point than point1! {value_z}")
                ret.status, ret.x, ret.fun = "ImprovementOfP1Found", z, value_z
            return ret
        ret.status = "Failed"
        return ret

    def find_path(self, point1, value_point1, point2, value_point2, f, MINR):
        ret = BasinsLoc.SearchResult()
        return self.dfs(point1, value_point1, point2, value_point2, f, ret, MINR)

    def add_point(self, point, value, component, ids, X, y):
        new_id = len(X)
        X.append(point)
        y.append(value)
        self.to.append(new_id)
        if component:
            component.append(new_id)
        ids.append(new_id)
        return new_id

    def delete_point(self, point_id, component, ids):
        ids.remove(point_id)
        if component:
            component.remove(point_id)

    def find_maximal_independent_set(self, component, X, y, r):
        values = np.zeros(len(component))
        for i in range(len(component)):
            values[i] = y[component[i]]
        sorted_inds = np.argsort(values)
        ind_set = []
        c = np.array(component)
        for i in c[sorted_inds]:
            include = True
            for j in ind_set:
                if BasinsLoc.dist(X[i], X[j]) < r:
                    include = False
                    break
            if include:
                ind_set.append(i)
        return ind_set

    def select_points(self, component, ids, X, y, known_pairs):
        perm = np.random.permutation(len(component))
        for i in perm:
            p1_id = component[i]
            min_dist, p2_id = float("inf"), None
            for p_id in component:
                d = BasinsLoc.dist(X[p_id], X[p1_id])
                if (p1_id, p_id) not in known_pairs and 0 < d < min_dist:
                    min_dist = d
                    p2_id = p_id
            if p2_id:
                if y[p2_id] < y[p1_id]:
                    return p2_id, p1_id
                return p1_id, p2_id
        return None, None

    def dump_points(self, ids, points, values):
        # with open(r"outputs/points_c_1.csv", "w") as f:
        #     for i in ids:
        #         print(values[i], *points[i], sep=",", file=f)
        if self.verbose:
            print("New points saved")

    def save_component_changes(self, c_old, c_new, X):
        with open("outputs/comp_changes.txt", "a") as f:
            print("{", end="", file=f)
            for i, ind in enumerate(c_old):
                print("[", end="", file=f)
                print(*X[ind], sep=",", end="", file=f)
                print("]", end="", file=f)
                if i != len(c_old) - 1:
                    print(",", end="", file=f)
            print("}", end=" -> ", file=f)
            print("{", end="", file=f)
            for i, ind in enumerate(c_new):
                print("[", end="", file=f)
                print(*X[ind], sep=",", end="", file=f)
                print("]", end="", file=f)
                if i != len(c_new) - 1:
                    print(",", end="", file=f)
            print("}", file=f)

    def save_path(self, rev_path, status):
        with open(r"outputs/paths.txt", "a") as f:
            print("{", end="", file=f)
            print("[", end="", file=f)
            print(*rev_path[len(rev_path) - 1], sep=",", end="", file=f)
            print("]", end="", file=f)
            for i in range(1, len(rev_path)):
                print(",[", end="", file=f)
                print(*rev_path[len(rev_path) - 1 - i], sep=",", end="", file=f)
                print("]", end="", file=f)
            if len(rev_path) > 0:
                print(",", end="", file=f)
            print(status, "}", sep="", file=f)

    def save_path_new_format(self, path, status):
        with open("paths.txt", "a") as f:
            print("{", end="", file=f)
            print("[", end="", file=f)
            print(*path[0], sep=",", end="", file=f)
            print("]", end="", file=f)
            for i in range(1, len(path)):
                print(",[", end="", file=f)
                print(*path[i], sep=",", end="", file=f)
                print("]", end="", file=f)
            if len(path) > 0:
                print(",", end="", file=f)
            print(status, "}", sep="", file=f)

    def compress_component(self, component, ids, X, y, f, r, MINR):
        known_pairs = set()
        if self.verbose:
            print(f"N points in component {len(component)}")
        while True:
            i, j = self.select_points(component, ids, X, y, known_pairs)
            if not i:
                break
            if self.verbose:
                print(f"From {y[j]} to {y[i]}")
            ret = self.find_path(X[j], y[j], X[i], y[i], f, MINR)
            if ret.status == "PathFound":
                self.delete_point(j, component, ids)
                self.delete_point(i, component, ids)
                self.add_point(ret.x, ret.fun, component, ids, X, y)
            elif ret.status == "ImprovementOfP1Found":
                self.delete_point(j, component, ids)
                k = self.add_point(ret.x, ret.fun, component, ids, X, y)
                known_pairs.add((k, i))
                known_pairs.add((i, k))
            elif ret.status == "ImprovementOfBothFound":
                self.delete_point(j, component, ids)
                self.add_point(ret.x, ret.fun, component, ids, X, y)
            elif ret.status == "Failed":
                known_pairs.add((i, j))
                known_pairs.add((j, i))
            if self.verbose:
                print(f"N points in component {len(component)}")
                print(f"N points {len(ids)}")
            self.dump_points(ids, X, y)
            if len(component) == 1:
                break
        return component

    def dfs1(self, u, X, c, used, dists, r):
        used[u] = True
        c.append(u)
        for i in range(len(X)):
            if not used[i] and 0 < dists[u][i] < r:
                self.dfs1(i, X, c, used, dists, r)

    def create_componenets(self, X, y, MINR):
        n = len(X)
        dists = np.zeros((n, n))
        components = []
        for i in range(n):
            for j in range(n):
                dists[i][j] = BasinsLoc.dist(X[i], X[j])
        used = [False] * n
        for i in range(n):
            if not used[i]:
                c = []
                self.dfs1(i, X, c, used, dists, MINR)
                components.append(c)
        components.sort(key=lambda l: len(l))
        return components

    def alg_closest_points(self, F, input_filename=None, X=None, y=None):
        if input_filename is not None:
            _X = np.genfromtxt(input_filename, delimiter=",")
            X, y = _X[:, 1:], _X[:, 0]
        else:
            X, y = X, y
        if self.verbose:
            print(X, y)
        X = [x for x in X]
        self.X = X
        y = y.tolist()
        ids = [i for i in range(len(X))]
        known_pairs = set()
        self.to = [i for i in range(len(X))]
        while True:
            mi, mi_arg = float("inf"), None
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    d = BasinsLoc.dist(X[ids[i]], X[ids[j]])
                    if (ids[i], ids[j]) not in known_pairs and d < mi:
                        mi = d
                        mi_arg = (ids[i], ids[j])
            if not mi_arg:
                return len(ids)
            i, j = mi_arg
            if y[i] > y[j]:
                i, j = j, i
            if self.verbose:
                print(f"From {y[j]} to {y[i]}")
            ret = self.find_path(X[j], y[j], X[i], y[i], F, self.MIN_DIST)
            if ret.status == "PathFound":
                self.delete_point(j, None, ids)
                self.delete_point(i, None, ids)
                k = self.add_point(ret.x, ret.fun, None, ids, X, y)
                self.to[j] = k
                self.to[i] = k
            elif ret.status == "ImprovementOfP1Found":
                self.delete_point(j, None, ids)
                k = self.add_point(ret.x, ret.fun, None, ids, X, y)
                known_pairs.add((k, i))
                known_pairs.add((i, k))
                self.to[j] = k
            elif ret.status == "ImprovementOfBothFound":
                self.delete_point(j, None, ids)
                k = self.add_point(ret.x, ret.fun, None, ids, X, y)
                self.to[j] = k
            elif ret.status == "Failed":
                known_pairs.add((i, j))
                known_pairs.add((j, i))
            #self.save_path(ret.path, ret.status)
            if self.verbose:
                print(f"N points {len(ids)}")
            self.dump_points(ids, X, y)

    def plot_attraction_basins(self, F, input_filename=None, X=None):
        if input_filename is not None:
            _X = np.genfromtxt(input_filename, delimiter=",")
            X_init = _X[:, 1:]
        else:
            X_init = X

        beg_to_end = np.zeros(len(self.X), dtype=int) - 1

        def dfs1(to, u):
            beg_to_end[u] = u
            v = to[u]
            if beg_to_end[v] == -1:
                dfs1(to, v)
            beg_to_end[u] = beg_to_end[v]

        fig, ax = utils.plot_3D_surface(
            (-5, 5), (-5, 5), F, X_init, is_colorbar=False, is_axis_names=False
        )
        zax_min, _ = ax.get_zlim()
        for i, x in enumerate(X_init):
            dfs1(self.to, i)
            start_point = x
            end_point = self.X[beg_to_end[i]]
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
        return fig, ax


if __name__ == "__main__":
    import argparse
    import os

    directory = "outputs"
    os.makedirs(directory, exist_ok=True)

    parser = argparse.ArgumentParser("Analyze Attration Basins")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the file with points to process",
    )
    parser.add_argument(
        "-f",
        "--function",
        required=True,
        type=str,
        help="Objective function",
    )
    args = parser.parse_args()
    if args.function == "rastrigin":
        F = utils.rastrigin
    elif args.function == "sphere":
        F = utils.sphere
    else:
        raise ValueError(f"Function {args.function} is not supported")
    bloc = BasinsLoc()
    n = bloc.alg_closest_points(F, input_filename=args.input)
    print(f"Number of attraction basins: {n}")
    fig, ax = bloc.plot_attraction_basins(F, input_filename=args.input)
    fig.savefig("outputs/after.pdf")
