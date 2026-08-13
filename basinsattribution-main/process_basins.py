import numpy as np
import os
import json
import analyze_basins
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import ioh
import signal

import sys

ROW_TIMEOUT_SECONDS = 6000


def timeout_handler(signum, frame):
    raise TimeoutError


def load_completed_ids(output_file):
    """Return IDs of completed rows and discard a crash-truncated final line."""
    if not os.path.exists(output_file):
        return set()

    completed_ids = set()
    last_valid_offset = 0

    with open(output_file, "rb") as f_out:
        while True:
            line = f_out.readline()
            if not line:
                break

            if not line.strip():
                last_valid_offset = f_out.tell()
                continue

            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Appending can only leave an incomplete record at the end.
                # Refuse to silently discard corruption in the middle.
                if f_out.read().strip():
                    raise ValueError(
                        f"Malformed JSONL record before the end of {output_file}"
                    )
                with open(output_file, "r+b") as repair_file:
                    repair_file.truncate(last_valid_offset)
                break

            if (
                row.get("id") is not None
                and "basin_info" in row
                and "nr_of_basins" in row
            ):
                completed_ids.add(row["id"])
            last_valid_offset = f_out.tell()

    return completed_ids

def process_bbob(fid=1):
    output_file = os.path.join("outputs", f"bbob.jsonl")
    n = 10
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)
    X1, X2 = np.meshgrid(x1, x2)
    X_data = np.column_stack([X1.ravel(), X2.ravel()])
    iid = 4
    #for iid in tqdm.tqdm(range(5), desc="instances", leave=False):

    row = {"function_id": fid, "instance_id": iid}
    F = ioh.get_problem(fid, iid, 2)
    
    y_data = np.array([F(x) for x in X_data])
    bloc = analyze_basins.BasinsLoc()
    nr_of_optima = bloc.alg_closest_points(F, X=X_data, y=y_data)

    def find_root(i, to):
        while to[i] != i:
            to[i] = to[to[i]]
            i = to[i]
        return i

    def collapse_to(to):
        for i in range(len(to)):
            to[i] = find_root(i, to)
        return to

    to = np.array(bloc.to).copy()
    roots = collapse_to(to)
    roots_init = roots[:len(X_data)]

    unique_basins, counts = np.unique(roots_init, return_counts=True)
    basin_info = [
        (int(size), bloc.X[basin_id].tolist(), float(F(bloc.X[basin_id])))
        for basin_id, size in zip(unique_basins, counts)
    ]

    row["basin_info"] = basin_info
    row["nr_of_basins"] = int(nr_of_optima)

    with open(output_file, "a") as f_out:
        json.dump(row, f_out)
        f_out.write("\n")

    return f"Done bbob f{fid}"


def process_experiment(exp_dir, base_dir):
    datadir = os.path.join(base_dir, exp_dir)
    output_file = os.path.join("outputs", f"{exp_dir}.jsonl")

    with open(f"{datadir}/log.jsonl", "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    completed_ids = load_completed_ids(output_file)
    eligible_rows = [row for row in data if row["fitness"] >= 0.5]
    remaining_rows = [
        row for row in eligible_rows if row.get("id") not in completed_ids
    ]

    if not remaining_rows:
        return f"Skipped {exp_dir} ({len(completed_ids)} rows already complete)"

    for row in tqdm.tqdm(remaining_rows, desc=exp_dir, leave=False):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(ROW_TIMEOUT_SECONDS)
        try:
            ns = {}
            exec(row["code"], ns)
            F = getattr(ns[row["name"]](dim=2), "f")

            n = 10
            x1 = np.linspace(-5, 5, n)
            x2 = np.linspace(-5, 5, n)
            X1, X2 = np.meshgrid(x1, x2)
            X_data = np.column_stack([X1.ravel(), X2.ravel()])
            y_data = np.array([F(x) for x in X_data])

            bloc = analyze_basins.BasinsLoc()
            nr_of_optima = bloc.alg_closest_points(F, X=X_data, y=y_data)

            def find_root(i, to):
                while to[i] != i:
                    to[i] = to[to[i]]
                    i = to[i]
                return i

            def collapse_to(to):
                for i in range(len(to)):
                    to[i] = find_root(i, to)
                return to

            to = np.array(bloc.to).copy()
            roots = collapse_to(to)
            roots_init = roots[:len(X_data)]

            unique_basins, counts = np.unique(roots_init, return_counts=True)
            basin_info = [
                (int(size), bloc.X[basin_id].tolist(), float(F(bloc.X[basin_id])))
                for basin_id, size in zip(unique_basins, counts)
            ]

            row["basin_info"] = basin_info
            row["nr_of_basins"] = int(nr_of_optima)
        except TimeoutError:
            tqdm.tqdm.write(
                f"Timed out {exp_dir} row {row.get('id')} "
                f"after {ROW_TIMEOUT_SECONDS} seconds"
            )
            continue
        finally:
            signal.alarm(0)

        # Write a complete JSONL record in one call and flush it so this row is
        # a durable checkpoint before starting the next expensive calculation.
        record = json.dumps(row) + "\n"
        with open(output_file, "a") as f_out:
            f_out.write(record)
            f_out.flush()
            os.fsync(f_out.fileno())

    return (
        f"Done {exp_dir} (processed {len(remaining_rows)}, "
        f"resumed past {len(completed_ids)})"
    )

if __name__ == "__main__":

    sys.setrecursionlimit(10000)

    os.makedirs("outputs", exist_ok=True)

    base_dir = "/local/bodasap/LLaMEA-ELA/exp_res_oai/"
    # get a list of folders in base_dir that start with "exp"
    experiment_dirs = [f for f in os.listdir(base_dir) if f.startswith("exp")]

    print(f"Found {len(experiment_dirs)} experiments to process.")

    if True:
        # one clean global progress bar
        with tqdm.tqdm(total=len(experiment_dirs), desc="Experiments", ncols=90) as pbar:
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_experiment, exp, base_dir): exp for exp in experiment_dirs}
                for fut in as_completed(futures):
                    msg = fut.result()
                    tqdm.tqdm.write(msg)
                    pbar.update(1)

    # Process BBOB functions[]
    #process_bbob(15)
    # with tqdm.tqdm(total=24, desc="BBOB Functions", ncols=90) as pbar:
    #     with ProcessPoolExecutor(max_workers=24) as executor:
    #         futures = {executor.submit(process_bbob, fid): fid for fid in range(1, 24 + 1)}
    #         for fut in as_completed(futures):
    #             msg = fut.result()
    #             tqdm.tqdm.write(msg)
    #             pbar.update(1)
