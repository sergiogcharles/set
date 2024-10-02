from nbody_gravity import GravitySim
import time
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt

"""
nbody: python -u generate_dataset.py  --num-train 50000 --sample-freq 500 2>&1 | tee log_generating_100000.log &

nbody_small: python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small 2>&1 | tee log_generating_10000_small.log &

"""


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulation", type=str, default="gravity", help="What simulation to generate."
)
parser.add_argument(
    "--num_total",
    type=int,
    default=100000,
    help="Total number of simulations to generate.",
)

parser.add_argument(
    "--percent_train",
    type=float,
    default=0.8,
    help="Percent of simulations to use as training data.",
)

parser.add_argument(
    "--percent_valid",
    type=float,
    default=0.1,
    help="Percent of simulations to use as cross-validation data.",
)

parser.add_argument(
    "--percent_test",
    type=float,
    default=0.1,
    help="Percent of simulations to use as test data.",
)

parser.add_argument("--seq_len", type=int, default=10, help="Length of sequence.")
parser.add_argument(
    "--horizon_len",
    type=int,
    default=1000,
    help="Length of horizon.",
)

parser.add_argument(
    "--sample-freq", type=int, default=20, help="How often to sample the trajectory."
)

parser.add_argument(
    "--num_masses", type=int, default=5, help="Number of massive particles/planets."
)
parser.add_argument("--tEnd", type=float, default=10, help="Ending timestep.")
parser.add_argument(
    "--dt", type=float, default=0.01, help="Time differential/timestep."
)
parser.add_argument("--softening", type=float, default=0.1, help="Softening length.")
parser.add_argument(
    "--G", type=float, default=1.0, help="Newton's Gravitational constant."
)

parser.add_argument(
    "--plot_real_time",
    type=str_to_bool,
    default=False,
    help="Whether to use spatial attention.",
)

parser.add_argument("--seed", type=int, default=3, help="Random seed.")
# parser.add_argument(
#     "--initial_vel", type=int, default=1, help="consider initial velocity"
# )
parser.add_argument("--sufix", type=str, default="", help="add a sufix to the name")

args = parser.parse_args()

# initial_vel_norm = 0.5
# if not args.initial_vel:
#     initial_vel_norm = 1e-16

if args.simulation == "gravity":
    sim = GravitySim(
        num_masses=args.num_masses,
        tEnd=args.tEnd,
        dt=args.dt,
        softening=args.softening,
        G=args.G,
        plot_real_time=args.plot_real_time,
    )
    suffix = "_gravity"
else:
    raise ValueError("Simulation {} not implemented".format(args.simulation))

# suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.sufix
suffix += str(args.num_masses) + str(args.sufix)
suffix += "_seqlen_" + str(args.seq_len)

print(suffix)


def generate_dataset(num_sims, seq_len, horizon_len, sample_freq):
    # total length = 1000 (T = seq_len * sample_freq = 100 * 10) + 1000 (H)
    length = seq_len * sample_freq + horizon_len
    print(
        f"Total time: {args.tEnd}, Timestep: {args.dt}, Total length: {length}, sequence length: {args.seq_len * args.sample_freq} @ sample freq: {args.sample_freq}"
    )
    loc_all = list()
    vel_all = list()
    edges_all = list()
    masses_all = list()
    for i in range(num_sims):
        t = time.time()
        loc, vel, edges, masses = sim.sample_trajectory(trajectory_len=length)

        # print(
        #     f"loc: {loc.shape} vel: {vel.shape} edges: {edges.shape} masses: {masses.shape}"
        # )
        # breakpoint()

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

        # Compress into correct format (num_examples, seq_len * sample_freq + 1, n, N) for n=3, N=5
        # We have +1 for the final step which we will predict
        # loc = np.concatenate(
        #     (
        #         loc[: seq_len * sample_freq : sample_freq, :, :],
        #         np.expand_dims(
        #             loc[seq_len * sample_freq + horizon_len - 1, :, :], axis=0
        #         ),
        #     ),
        #     axis=0,
        # )
        loc = np.concatenate(
            (
                loc[: seq_len * sample_freq : sample_freq, :, :],
                np.expand_dims(loc[-1, :, :], axis=0),
            ),
            axis=0,
        )

        # vel = np.concatenate(
        #     (
        #         vel[: seq_len * sample_freq : sample_freq, :, :],
        #         np.expand_dims(
        #             vel[seq_len * sample_freq + horizon_len - 1, :, :], axis=0
        #         ),
        #     ),
        #     axis=0,
        # )
        vel = np.concatenate(
            (
                vel[: seq_len * sample_freq : sample_freq, :, :],
                np.expand_dims(vel[-1, :, :], axis=0),
            ),
            axis=0,
        )

        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        masses_all.append(masses)

    masses_all = np.stack(masses_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    print(
        f"masses {masses_all.shape}, loc {loc_all.shape}, vel {vel_all.shape}, edges {edges_all.shape}"
    )

    return loc_all, vel_all, edges_all, masses_all


if __name__ == "__main__":
    np.random.seed(args.seed)

    assert args.percent_train + args.percent_valid + args.percent_test == 1.0

    print("Generating {} simulations".format(args.num_total))
    loc, vel, edges, masses = generate_dataset(
        args.num_total, args.seq_len, args.horizon_len, args.sample_freq
    )
    # print(f'loc {loc.shape}, vel {vel.shape}, edges {edges.shape}, masses {masses.shape}')

    num_train = math.floor(args.percent_train * args.num_total)
    num_valid = math.floor(args.percent_valid * args.num_total)
    num_test = math.floor(args.percent_test * args.num_total)

    loc_train, loc_valid, loc_test = (
        loc[:num_train, :, :, :],
        loc[num_train : num_train + num_valid, :, :, :],
        loc[num_train + num_valid :, :, :, :],
    )
    vel_train, vel_valid, vel_test = (
        vel[:num_train, :, :, :],
        vel[num_train : num_train + num_valid, :, :, :],
        vel[num_train + num_valid :, :, :, :],
    )
    edges_train, edges_valid, edges_test = (
        edges[:num_train, :, :],
        edges[num_train : num_train + num_valid, :, :],
        edges[num_train + num_valid :, :, :],
    )
    masses_train, masses_valid, masses_test = (
        masses[:num_train, :, :],
        masses[num_train : num_train + num_valid, :, :],
        masses[num_train + num_valid :, :, :],
    )

    print(
        f"loc_train {loc_train.shape}, vel_train {vel_train.shape}, edges_train {edges_train.shape}, masses_train {masses_train.shape}"
    )
    print(
        f"loc_valid {loc_valid.shape}, vel_valid {vel_valid.shape}, edges_valid {edges_valid.shape}, masses_valid {masses_valid.shape}"
    )
    print(
        f"loc_test {loc_test.shape}, vel_test {vel_test.shape}, edges_test {edges_test.shape}, masses_test {masses_test.shape}"
    )

    # print("Generating {} training simulations".format(args.num_train))
    # loc_train, vel_train, edges_train, masses_train = generate_dataset(
    #     args.num_train, args.train_seq_len, args.train_horizon_len, args.sample_freq
    # )

    # print("Generating {} validation simulations".format(args.num_valid))
    # loc_valid, vel_valid, edges_valid, masses_valid = generate_dataset(
    #     args.num_valid, args.train_seq_len, args.train_horizon_len, args.sample_freq
    # )

    # print("Generating {} test simulations".format(args.num_test))
    # loc_test, vel_test, edges_test, masses_test = generate_dataset(
    #     args.num_test, args.test_seq_len, args.test_horizon_len, args.sample_freq
    # )

    np.save("loc_train" + suffix + ".npy", loc_train)
    np.save("vel_train" + suffix + ".npy", vel_train)
    np.save("edges_train" + suffix + ".npy", edges_train)
    np.save("masses_train" + suffix + ".npy", masses_train)

    np.save("loc_valid" + suffix + ".npy", loc_valid)
    np.save("vel_valid" + suffix + ".npy", vel_valid)
    np.save("edges_valid" + suffix + ".npy", edges_valid)
    np.save("masses_valid" + suffix + ".npy", masses_valid)

    np.save("loc_test" + suffix + ".npy", loc_test)
    np.save("vel_test" + suffix + ".npy", vel_test)
    np.save("edges_test" + suffix + ".npy", edges_test)
    np.save("masses_test" + suffix + ".npy", masses_test)
