import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.trial import TrialState
import datetime

from n_body_system.model import (
    GNN,
    EGNN,
    Baseline,
    Linear,
    EGNN_vel,
    MLP_dynamics,
    RF_vel,
)
import os
from torch import nn, optim
import json
import time
from torch.autograd import Variable

torch.manual_seed(0)


parser = argparse.ArgumentParser(
    description="MLP dynamics Baseline Trajectory Prediction Example"
)

parser.add_argument(
    "--exp_name", type=str, default="mlp_dynamics_baseline", help="experiment_name"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="input batch size for training (default: 128)",
)

parser.add_argument(
    "--train_seq_len",
    type=int,
    default=100,
    help="sequence length during training.",
)

parser.add_argument(
    "--train_horizon_len",
    type=int,
    default=5000,
    help="horizon length during training.",
)

parser.add_argument(
    "--test_seq_len",
    type=int,
    default=100,
    help="sequence length during testing.",
)

parser.add_argument(
    "--test_horizon_len",
    type=int,
    default=5000,
    help="horizon length during testing.",
)

parser.add_argument(
    "--input_nf",
    type=int,
    default=3,
    help="input dimension",
)

parser.add_argument(
    "--hidden_nf",
    type=int,
    default=1024,
    help="hidden dimension",
)

parser.add_argument(
    "--output_nf",
    type=int,
    default=3,
    help="output dimension",
)

parser.add_argument(
    "--weight_decay", type=float, default=1e-12, help="timing experiment"
)

parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")

parser.add_argument("--clip", type=float, default=10, help="gradient clipping")

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="number of epochs to train (default: 10)",
)

parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

parser.add_argument(
    "--outf",
    type=str,
    default="n_body_system/logs",
    help="folder to output vae",
)

parser.add_argument(
    "--max_training_samples",
    type=int,
    default=3000,
    help="maximum amount of training samples",
)
parser.add_argument(
    "--dataset", type=str, default="nbody_small", metavar="N", help="nbody_small, nbody"
)
parser.add_argument("--time_exp", type=int, default=0, help="timing experiment")

parser.add_argument("--norm_diff", type=eval, default=False, help="normalize_diff")

parser.add_argument("--recurrent", type=eval, default=True, help="use recurrent")

parser.add_argument("--tanh", type=eval, default=False, help="use tanh")

parser.add_argument("--tensorboard", type=bool, default=False, help="Uses tensorboard")

parser.add_argument(
    "--adjacency",
    type=bool,
    default=True,
    help="Whether to use adjacency matrix/edge attribute self-attention.",
)

parser.add_argument(
    "--time_series",
    type=bool,
    default=False,
    help="Whether to use all time series data.",
)

parser.add_argument(
    "--optuna",
    type=bool,
    default=False,
    help="Whether to perform HPO with Optuna.",
)


time_exp_dic = {"time": 0, "counter": 0}

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def create_summary_writer(lr, time_series):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_lr_{lr}_ts_{time_series}/"

    writer = SummaryWriter(log_dir)
    return writer


def objective(trial):
    dataset_train = NBodyDataset(
        partition="train",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyDataset(
        partition="val",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name="nbody_small",
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    dataset_test = NBodyDataset(
        partition="test",
        seq_len=args.test_seq_len,
        horizon_len=args.test_horizon_len,
        dataset_name="nbody_small",
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    input_nf = args.input_nf
    hidden_nf = args.hidden_nf
    output_nf = args.output_nf
    weight_decay = args.weight_decay

    # Hyperparameters for Optuna search
    lr = trial.suggest_float("lr", 1e-6, 6e-5, log=True)
    time_series = trial.suggest_categorical("time_series", [True, False])

    model = MLP_dynamics(input_nf=input_nf, hidden_nf=hidden_nf, output_nf=output_nf)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(lr, time_series)

    # early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0.1)

    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train, time_series)
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # if epoch % args.test_interval == 0:
        val_loss = val(
            model,
            epoch,
            loader_val,
            time_series,
        )
        test_loss = val(
            model,
            epoch,
            loader_test,
            time_series,
        )

        if args.tensorboard:
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

        results["epochs"].append(epoch)
        results["losess"].append(test_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_epoch = epoch
            # save model
            torch.save(model.state_dict(), "best_models/optuna_mlp_dynamics_baseline.pt")

        print(
            "*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_val_loss, best_test_loss, best_epoch)
        )

        # json_object = json.dumps(results, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)
        trial.report(val_loss, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune() or early_stopper.early_stop(val_loss):
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main_optuna():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():
    dataset_train = NBodyDataset(
        partition="train",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyDataset(
        partition="val",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name="nbody_small",
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    dataset_test = NBodyDataset(
        partition="test",
        seq_len=args.test_seq_len,
        horizon_len=args.test_horizon_len,
        dataset_name="nbody_small",
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # Hyperparameters
    input_nf = args.input_nf
    hidden_nf = args.hidden_nf
    output_nf = args.output_nf
    time_series = args.time_series
    lr = args.lr
    weight_decay = args.weight_decay

    model = MLP_dynamics(input_nf=input_nf, hidden_nf=hidden_nf, output_nf=output_nf)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(lr, time_series)

    # early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0.1)

    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train, time_series)
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # if epoch % args.test_interval == 0:
        val_loss = val(model, epoch, loader_val, time_series)
        test_loss = val(model, epoch, loader_test, time_series)

        if args.tensorboard:
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

        results["epochs"].append(epoch)
        results["losess"].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_epoch = epoch
            # save model
            torch.save(model.state_dict(), "best_models/mlp_dynamics_baseline.pt")
        print(
            "*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_val_loss, best_test_loss, best_epoch)
        )

        # json_object = json.dumps(results, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)
        if early_stopper.early_stop(val_loss):
            break

    return best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, time_series):
    model.train()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    for batch_idx, data in enumerate(loader):
        # B = batch_size, N = n_nodes, L = seq_len, n = 3
        B, N, L, n = data[0].size()
        data = [d.to(device) for d in data]
        # loc, vel has dim (B, N, L, n) where n=3
        # loc_end, vel_end has dim (B, N, n)
        loc, vel, edge_attr, charges, loc_end, vel_end = data

        # print(
        #     f"loc: {loc.shape}, vel: {vel.shape}, loc_end {loc_end.shape}, vel_end {vel_end.shape}"
        # )

        # If we are using all of the time series from t=1,...,L
        if time_series:
            # loc, vel are now (B, N, L, n)
            loc = loc.reshape(B, N, L, n)
            vel = vel.reshape(B, N, L, n)

            optimizer.zero_grad()

            if args.time_exp:
                torch.cuda.synchronize()
                t1 = time.time()

            # MLP dynamics baseline
            loc_spatial = model(loc.detach(), vel.detach())

            # take the mean across time dimension
            loc_spatial = loc_spatial.mean(-2)
        else:
            # loc, vel are now (B, N, L, n)
            loc = loc.reshape(B, N, L, n)
            vel = vel.reshape(B, N, L, n)

            # take only the last entry at t=L
            # (B, N, n)
            loc = loc[:, :, -1, :]
            vel = vel[:, :, -1, :]

            optimizer.zero_grad()

            if args.time_exp:
                torch.cuda.synchronize()
                t1 = time.time()

            # MLP dynamics baseline
            loc_spatial = model(loc.detach(), vel.detach())

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic["time"] += t2 - t1
            time_exp_dic["counter"] += 1

            print(
                "Forward average time: %.6f"
                % (time_exp_dic["time"] / time_exp_dic["counter"])
            )

        # loss = (
        #     loss_mse(h_temporal_pred, h_temporal_end)
        #     + loss_mse(loc_temporal_pred, loc_end)
        #     + loss_mse(vel_temporal_pred, vel_end)
        # )

        loss = loss_mse(loc_spatial, loc_end)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        res["loss"] += loss.item() * B
        res["counter"] += B

    prefix = ""
    print(
        "%s epoch %d avg loss: %.5f"
        % (prefix + loader.dataset.partition, epoch, res["loss"] / res["counter"])
    )

    return res["loss"] / res["counter"]


def val(model, epoch, loader, time_series):
    model.eval()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # B = batch_size, N = n_nodes, L = seq_len, n = 3
            B, N, L, n = data[0].size()
            data = [d.to(device) for d in data]
            # loc, vel has dim (B, N, L, n) where n=3
            # loc_end, vel_end has dim (B, N, n)
            loc, vel, edge_attr, charges, loc_end, vel_end = data

            # print(
            #     f"loc: {loc.shape}, vel: {vel.shape}, loc_end {loc_end.shape}, vel_end {vel_end.shape}"
            # )

            # If we are using all of the time series from t=1,...,L
            if time_series:
                # loc, vel are now (B, N, L, n)
                loc = loc.reshape(B, N, L, n)
                vel = vel.reshape(B, N, L, n)

                if args.time_exp:
                    torch.cuda.synchronize()
                    t1 = time.time()

                # MLP dynamics baseline
                loc_spatial = model(loc.detach(), vel.detach())

                # take the mean across time dimension
                loc_spatial = loc_spatial.mean(-2)
            else:
                # loc, vel are now (B, N, L, n)
                loc = loc.reshape(B, N, L, n)
                vel = vel.reshape(B, N, L, n)

                # take only the last entry at t=L
                # (B, N, n)
                loc = loc[:, :, -1, :]
                vel = vel[:, :, -1, :]

                if args.time_exp:
                    torch.cuda.synchronize()
                    t1 = time.time()

                # MLP dynamics baseline
                loc_spatial = model(loc.detach(), vel.detach())

            if args.time_exp:
                torch.cuda.synchronize()
                t2 = time.time()
                time_exp_dic["time"] += t2 - t1
                time_exp_dic["counter"] += 1

                print(
                    "Forward average time: %.6f"
                    % (time_exp_dic["time"] / time_exp_dic["counter"])
                )

            # loss = (
            #     loss_mse(h_temporal_pred, h_temporal_end)
            #     + loss_mse(loc_temporal_pred, loc_end)
            #     + loss_mse(vel_temporal_pred, vel_end)
            # )

            loss = loss_mse(loc_spatial, loc_end)
            res["loss"] += loss.item() * B
            res["counter"] += B

    prefix = "==> "
    print(
        "%s epoch %d avg loss: %.5f"
        % (prefix + loader.dataset.partition, epoch, res["loss"] / res["counter"])
    )

    return res["loss"] / res["counter"]


if __name__ == "__main__":
    if args.optuna:
        main_optuna()
    else:
        best_val_loss, best_test_loss, best_epoch = main()
