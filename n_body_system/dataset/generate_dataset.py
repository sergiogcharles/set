from synthetic_sim import ChargedParticlesSim
import time
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt

"""
nbody: python -u generate_dataset.py  --num-train 50000 --sample-freq 500 2>&1 | tee log_generating_100000.log &

nbody_small: python -u generate_dataset.py --num-train 10000 --seed 43 --sufix small 2>&1 | tee log_generating_10000_small.log &

"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--simulation", type=str, default="charged", help="What simulation to generate."
)
parser.add_argument(
    "--num_total",
    type=int,
    default=50000,
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

# parser.add_argument(
#     "--num-train",
#     type=int,
#     default=10000,
#     help="Number of training simulations to generate.",
# )
# parser.add_argument(
#     "--num-valid",
#     type=int,
#     default=2000,
#     help="Number of validation simulations to generate.",
# )
# parser.add_argument(
#     "--num-test", type=int, default=2000, help="Number of test simulations to generate."
# )
parser.add_argument(
    "--train_seq_len", type=int, default=10, help="Length of sequence during training."
)
parser.add_argument(
    "--train_horizon_len",
    type=int,
    default=19900,
    help="Length of horizon during training.",
)
parser.add_argument(
    "--test_seq_len", type=int, default=10, help="Length of sequence during training."
)
parser.add_argument(
    "--test_horizon_len",
    type=int,
    default=19000,
    help="Length of horizon during testing",
)

parser.add_argument(
    "--sample-freq", type=int, default=10, help="How often to sample the trajectory."
)

parser.add_argument(
    "--n_balls", type=int, default=10, help="Number of balls in the simulation."
)
parser.add_argument("--seed", type=int, default=3, help="Random seed.")
parser.add_argument(
    "--initial_vel", type=int, default=1, help="consider initial velocity"
)
parser.add_argument("--sufix", type=str, default="", help="add a sufix to the name")
parser.add_argument(
    "--noise_var",
    type=float,
    default=0.0,
    help="Noise variance.",
)

args = parser.parse_args()

initial_vel_norm = 0.5
if not args.initial_vel:
    initial_vel_norm = 1e-16

if args.simulation == "charged":
    # sim = ChargedParticlesSim(
    #     noise_var=args.noise_var, n_balls=args.n_balls, vel_norm=initial_vel_norm
    # )
    sim = ChargedParticlesSim(n_balls=args.n_balls, loc_std=2)
    suffix = "_charged"
else:
    raise ValueError("Simulation {} not implemented".format(args.simulation))

suffix += str(args.n_balls) + "_initvel%d" % args.initial_vel + args.sufix

print(suffix)


def generate_dataset(num_sims, seq_len, horizon_len, sample_freq):
    # total length = 1000 (T = seq_len * sample_freq = 100 * 10) + 1000 (H)
    length = seq_len * sample_freq + horizon_len
    print(
        f"Total length: {length}, sequence length: {seq_len}, horizon length: {horizon_len}"
    )
    loc_all = list()
    vel_all = list()
    edges_all = list()
    charges_all = list()
    for i in range(num_sims):
        t = time.time()
        loc, vel, edges, charges = sim.sample_trajectory(
            T=length, sample_freq=sample_freq
        )

        # Plot
        # plt.figure()
        # axes = plt.gca()
        # axes.set_xlim([-10.0, 10.0])
        # axes.set_ylim([-10.0, 10.0])
        # for i in range(loc.shape[-1]):
        #     plt.plot(loc[:, 0, i], loc[:, 1, i])
        #     plt.plot(loc[0, 0, i], loc[0, 1, i], "d")
        # plt.figure()
        # plt.show()
        # breakpoint()
        # print(f"loc: {loc.shape} {vel.shape} {edges.shape} {charges.shape}")

        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))

        # Compress into correct format (num_examples, seq_len * sample_freq + 1, n, N) for n=3, N=5
        # We have +1 for the final step which we will predict
        loc = np.concatenate(
            (
                loc[: seq_len * sample_freq : sample_freq, :, :],
                np.expand_dims(
                    loc[seq_len * sample_freq + horizon_len - 1, :, :], axis=0
                ),
            ),
            axis=0,
        )

        vel = np.concatenate(
            (
                vel[: seq_len * sample_freq : sample_freq, :, :],
                np.expand_dims(
                    vel[seq_len * sample_freq + horizon_len - 1, :, :], axis=0
                ),
            ),
            axis=0,
        )

        # print(f"loc: {loc.shape} {vel.shape} {edges.shape} {charges.shape}")
        # breakpoint()

        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)
        charges_all.append(charges)

    charges_all = np.stack(charges_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    print(
        f"charges {charges_all.shape}, loc {loc_all.shape}, vel {vel_all.shape}, edges {edges_all.shape}"
    )

    return loc_all, vel_all, edges_all, charges_all


if __name__ == "__main__":
    np.random.seed(args.seed)

    assert args.percent_train + args.percent_valid + args.percent_test == 1.0

    print("Generating {} simulations".format(args.num_total))
    loc, vel, edges, charges = generate_dataset(
        args.num_total, args.train_seq_len, args.train_horizon_len, args.sample_freq
    )
    # print(f'loc {loc.shape}, vel {vel.shape}, edges {edges.shape}, charges {charges.shape}')

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
    charges_train, charges_valid, charges_test = (
        charges[:num_train, :, :],
        charges[num_train : num_train + num_valid, :, :],
        charges[num_train + num_valid :, :, :],
    )

    print(
        f"loc_train {loc_train.shape}, vel_train {vel_train.shape}, edges_train {edges_train.shape}, charges_train {charges_train.shape}"
    )
    print(
        f"loc_valid {loc_valid.shape}, vel_valid {vel_valid.shape}, edges_valid {edges_valid.shape}, charges_valid {charges_valid.shape}"
    )
    print(
        f"loc_test {loc_test.shape}, vel_test {vel_test.shape}, edges_test {edges_test.shape}, charges_test {charges_test.shape}"
    )

    # print("Generating {} training simulations".format(args.num_train))
    # loc_train, vel_train, edges_train, charges_train = generate_dataset(
    #     args.num_train, args.train_seq_len, args.train_horizon_len, args.sample_freq
    # )

    # print("Generating {} validation simulations".format(args.num_valid))
    # loc_valid, vel_valid, edges_valid, charges_valid = generate_dataset(
    #     args.num_valid, args.train_seq_len, args.train_horizon_len, args.sample_freq
    # )

    # print("Generating {} test simulations".format(args.num_test))
    # loc_test, vel_test, edges_test, charges_test = generate_dataset(
    #     args.num_test, args.test_seq_len, args.test_horizon_len, args.sample_freq
    # )

    np.save("loc_train" + suffix + ".npy", loc_train)
    np.save("vel_train" + suffix + ".npy", vel_train)
    np.save("edges_train" + suffix + ".npy", edges_train)
    np.save("charges_train" + suffix + ".npy", charges_train)

    np.save("loc_valid" + suffix + ".npy", loc_valid)
    np.save("vel_valid" + suffix + ".npy", vel_valid)
    np.save("edges_valid" + suffix + ".npy", edges_valid)
    np.save("charges_valid" + suffix + ".npy", charges_valid)

    np.save("loc_test" + suffix + ".npy", loc_test)
    np.save("vel_test" + suffix + ".npy", vel_test)
    np.save("edges_test" + suffix + ".npy", edges_test)
    np.save("charges_test" + suffix + ".npy", charges_test)
