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
    Linear_dynamics,
    RF_vel,
    EGNNBaseline,
)
import os
from torch import nn, optim
import json
import time

torch.manual_seed(0)


parser = argparse.ArgumentParser(
    description="EGNN_vel Baseline Trajectory Prediction Example"
)


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


parser.add_argument(
    "--exp_name", type=str, default="egnn_vel_baseline", help="experiment_name"
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
    default=10,
    help="sequence length during training.",
)

parser.add_argument(
    "--train_horizon_len",
    type=int,
    default=10000,
    help="horizon length during training.",
)

parser.add_argument(
    "--test_seq_len",
    type=int,
    default=10,
    help="sequence length during testing.",
)

parser.add_argument(
    "--test_horizon_len",
    type=int,
    default=10000,
    help="horizon length during testing.",
)

parser.add_argument(
    "--sample_freq",
    type=int,
    default=1,
    help="How frequently to sample sequence data for model.",
)

parser.add_argument(
    "--num_atoms",
    type=int,
    default=5,
    help="number of atoms",
)

parser.add_argument(
    "--in_node_nf",
    type=int,
    default=1,
    help="dimension of input node features h",
)

parser.add_argument(
    "--in_edge_nf",
    type=int,
    default=2,
    help="dimension of input edge features e",
)

parser.add_argument(
    "--in_pos_nf",
    type=int,
    default=3,
    help="dimension of input coordinates",
)

parser.add_argument(
    "--hidden_nf",
    type=int,
    default=64,
    help="hidden dimension",
)

parser.add_argument(
    "--proj_nf",
    type=int,
    default=1,
    help="projection dimension",
)

parser.add_argument(
    "--n_egnn_layers",
    type=int,
    default=4,
    help="number of EGNN layers",
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
    "--log_interval",
    type=int,
    default=1,
    help="how many batches to wait before logging training status",
)

parser.add_argument(
    "--test_interval",
    type=int,
    default=5,
    help="how many epochs to wait before logging test",
)

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

parser.add_argument(
    "--tensorboard", type=str_to_bool, default=False, help="Uses tensorboard"
)

parser.add_argument(
    "--time_series",
    type=str_to_bool,
    default=False,
    help="Whether to use all time series data.",
)

parser.add_argument(
    "--optuna",
    type=str_to_bool,
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
    def __init__(
        self, patience=1, min_delta=0, plateau_patience=5, plateau_threshold=1e-3
    ):
        self.patience = patience
        self.plateau_patience = plateau_patience

        self.min_delta = min_delta
        self.plateau_threshold = plateau_threshold

        self.counter = 0
        self.plateau_counter = 0

        self.min_validation_loss = float("inf")
        self.last_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        # Check for plateau
        # if abs(validation_loss - self.last_validation_loss) < self.plateau_threshold:
        #     self.plateau_counter += 1
        #     if self.plateau_counter >= self.plateau_patience:
        #         return True
        # else:
        #     self.plateau_counter = 0

        # self.last_validation_loss = validation_loss

        return False


def create_summary_writer(
    lr,
    weight_decay,
    time_series,
    batch_size,
    hidden_nf,
    recurrent,
    n_egnn_layers,
    optuna=False,
):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_egnn_vel_baseline_optuna_{optuna}_lr_{lr}_wd_{weight_decay}_ts_{time_series}_bsz_{batch_size}_hnf_{hidden_nf}_recurrent_{recurrent}_n_egnn_layers_{n_egnn_layers}/"

    writer = SummaryWriter(log_dir)
    return writer


def objective(trial):
    # Hyperparameters
    # batch_size = 100
    # train_seq_len = 100
    # num_atoms = 5
    # in_node_nf = 1
    # in_edge_nf = 2
    # in_pos_nf = 3
    # hidden_nf = 64
    # proj_nf = 1
    # n_egnn_layers = 4
    # recurrent = True
    # norm_diff = False
    # tanh = False
    # weight_decay = 1e-12

    # batch_size = args.batch_size
    train_seq_len = args.train_seq_len
    num_atoms = args.num_atoms
    in_node_nf = args.in_node_nf
    in_edge_nf = args.in_edge_nf
    in_pos_nf = args.in_pos_nf
    # hidden_nf = args.hidden_nf
    proj_nf = args.proj_nf
    # n_egnn_layers = args.n_egnn_layers
    # recurrent = args.recurrent
    norm_diff = args.norm_diff
    tanh = args.tanh
    # weight_decay = args.weight_decay

    # Hyperparameters for Optuna search
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-12, 1e-6])
    time_series = trial.suggest_categorical("time_series", [False, True])
    batch_size = trial.suggest_categorical("batch_size", [32, 100, 256])
    hidden_nf = trial.suggest_categorical("hidden_nf", [64, 256, 512])
    recurrent = trial.suggest_categorical("recurrent", [True, False])
    n_egnn_layers = trial.suggest_categorical("n_egnn_layers", [1, 2, 3])

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Dataset
    dataset_train = NBodyDataset(
        partition="train",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
        sample_freq=args.sample_freq,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyDataset(
        partition="val",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, drop_last=True
    )

    dataset_test = NBodyDataset(
        partition="test",
        seq_len=args.test_seq_len,
        horizon_len=args.test_horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
    )

    # Model
    model = EGNNBaseline(
        batch_size=batch_size,
        seq_len=train_seq_len,
        num_atoms=num_atoms,
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        in_pos_nf=in_pos_nf,
        hidden_nf=hidden_nf,
        proj_nf=proj_nf,
        time_series=time_series,
        device=device,
        n_egnn_layers=n_egnn_layers,
        recurrent=recurrent,
        norm_diff=norm_diff,
        tanh=tanh,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(
            lr,
            weight_decay,
            time_series,
            batch_size,
            hidden_nf,
            recurrent,
            n_egnn_layers,
            optuna=True,
        )

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
            torch.save(
                model.state_dict(),
                f"best_models/egnn_vel_baseline_optuna_{True}_lr_{lr}_wd_{weight_decay}_ts_{time_series}_bsz_{batch_size}_hnf_{hidden_nf}_recurrent_{recurrent}_n_egnn_layers_{n_egnn_layers}.pt",
            )

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
            if trial.should_prune():
                print(f"PRUNED")
            else:
                print(f"EARLY STOPPED")
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def main_optuna():
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        direction="minimize",
    )
    study.optimize(objective, n_trials=10)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Val Loss Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best hparam settings
    text_out = f"Val Loss Value: {trial.value}\n"
    text_out += " Params: \n"
    for key, value in trial.params.items():
        text_out += f"   {key}: {value}\n"
    file_name = f"optuna_study_egnn_vel_baseline.txt"
    with open(file_name, "w") as file:
        file.write(text_out)


def main():
    dataset_train = NBodyDataset(
        partition="train",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
        sample_freq=args.sample_freq,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyDataset(
        partition="val",
        seq_len=args.train_seq_len,
        horizon_len=args.train_horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    dataset_test = NBodyDataset(
        partition="test",
        seq_len=args.test_seq_len,
        horizon_len=args.test_horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # Hyperparameters
    batch_size = args.batch_size
    train_seq_len = args.train_seq_len
    num_atoms = args.num_atoms
    in_node_nf = args.in_node_nf
    in_edge_nf = args.in_edge_nf
    in_pos_nf = args.in_pos_nf
    hidden_nf = args.hidden_nf
    proj_nf = args.proj_nf
    time_series = args.time_series
    n_egnn_layers = args.n_egnn_layers
    recurrent = args.recurrent
    norm_diff = args.norm_diff
    tanh = args.tanh
    weight_decay = args.weight_decay
    lr = args.lr

    model = EGNNBaseline(
        batch_size=batch_size,
        seq_len=train_seq_len,
        num_atoms=num_atoms,
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        in_pos_nf=in_pos_nf,
        hidden_nf=hidden_nf,
        proj_nf=proj_nf,
        time_series=time_series,
        device=device,
        n_egnn_layers=n_egnn_layers,
        recurrent=recurrent,
        norm_diff=norm_diff,
        tanh=tanh,
    )

    total_params = sum(p.numel() for p in model.parameters())

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(
            lr,
            weight_decay,
            time_series,
            batch_size,
            hidden_nf,
            recurrent,
            n_egnn_layers,
            optuna=False,
        )

    # early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0.1)

    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_train_loss = float("inf")
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
            best_train_loss = train_loss
            best_epoch = epoch
            # save model
            torch.save(
                model.state_dict(),
                f"best_models/egnn_vel_baseline_optuna_{False}_lr_{lr}_wd_{weight_decay}_ts_{time_series}_bsz_{batch_size}_hnf_{hidden_nf}_recurrent_{recurrent}_n_egnn_layers_{n_egnn_layers}_{args.dataset}.pt",
            )

        print(
            "*** Best Train Loss: %.5f  Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_train_loss, best_val_loss, best_test_loss, best_epoch)
        )

        # json_object = json.dumps(results, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)
        if early_stopper.early_stop(val_loss):
            print(f"EARLY STOPPED")
            break

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params


def train(model, optimizer, epoch, loader, time_series):
    model.train()
    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    for batch_idx, data in enumerate(loader):
        # B = batch_size, N = n_nodes, L = seq_len, n = 3
        B, N, L, n = data[0].size()
        # print(f"train {B}, {N}, {L}, {n}")
        data = [d.to(device) for d in data]
        # loc, vel has dim (B, N, L, n) where n=3
        # loc_end, vel_end has dim (B, N, n)
        loc, vel, edge_attr, charges, loc_end, vel_end = data

        # print(
        #     f"loc: {loc.shape}, vel: {vel.shape}, loc_end {loc_end.shape}, vel_end {vel_end.shape}"
        # )

        # If we are using all of the time series from t=1,...,L
        if time_series:
            edges = loader.dataset.get_edges(B, N, L)
            edges = [edges[0].to(device), edges[1].to(device)]

            # generate features (vector norm) with dim (B, N, d) where d=1
            h_spatiotemporal_end = torch.norm(vel_end, p=2, dim=-1).unsqueeze(-1)

            # loc, vel are now (B * N * L, n)
            loc = loc.reshape(B * N * L, n)
            vel = vel.reshape(B * N * L, n)

            # h (or nodes) = ||v|| which is (B * N * L, d)
            h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
            rows, cols = edges

            # (B * N * (N - 1) * L, 1)
            loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
                -1
            )  # relative distances among locations

            # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
            # where N(N - 1) is the num of connections in the graph
            edge_attr = edge_attr.view(B * N * (N - 1), 1)
            # (B * N * (N - 1), 1) -> (B * N * (N - 1), 1, 1)
            edge_attr = edge_attr.unsqueeze(1)
            # (B * N * (N - 1), 1, 1) -> (B * N * (N - 1), L, 1)
            edge_attr = edge_attr.expand(-1, L, -1)
            # (B * N * (N - 1), L, 1) -> (B * N * (N - 1) * L, 1)
            edge_attr = edge_attr.reshape(B * N * (N - 1) * L, 1)

            # (B * N * (N - 1) * L, 2)
            edge_attr = torch.cat(
                [edge_attr, loc_dist], -1
            ).detach()  # concatenate all edge properties

            optimizer.zero_grad()

            if args.time_exp:
                torch.cuda.synchronize()
                t1 = time.time()

            # EGNN_vel Baseline
            # h_spatial is (B, N, d)
            # loc_spatial is (B, N, n)
            # vel_spatial is (B, N, n)
            edges[0], edges[1] = edges[0].detach(), edges[1].detach()
            (h_spatial, loc_spatial, vel_spatial) = model(
                h, loc.detach(), edges, vel.detach(), edge_attr
            )
        else:
            edges = loader.dataset.get_edges(B, N, 1)
            edges = [edges[0].to(device), edges[1].to(device)]

            # generate features (vector norm) with dim (B, N, d) where d=1
            h_spatial_end = torch.norm(vel_end, p=2, dim=-1).unsqueeze(-1)

            # loc, vel are (B, N, L, n)
            # take only the last entry at t=L
            loc = loc[:, :, -1, :]
            vel = vel[:, :, -1, :]

            # loc, vel are now (B * N, n)
            loc = loc.reshape(B * N, n)
            vel = vel.reshape(B * N, n)

            # h (or nodes) = ||v|| which is (B * N, d)
            h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
            rows, cols = edges

            # (B * N * (N - 1), 1)
            loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
                -1
            )  # relative distances among locations

            # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
            # where N(N - 1) is the num of connections in the graph
            edge_attr = edge_attr.view(B * N * (N - 1), 1)

            # (B * N * (N - 1), 2)
            edge_attr = torch.cat(
                [edge_attr, loc_dist], -1
            ).detach()  # concatenate all edge properties

            optimizer.zero_grad()

            if args.time_exp:
                torch.cuda.synchronize()
                t1 = time.time()

            # EGNN_vel Baseline
            # h_spatial is (B, N, d)
            # loc_spatial is (B, N, n)
            # vel_spatial is (B, N, n)
            edges[0], edges[1] = edges[0].detach(), edges[1].detach()
            (h_spatial, loc_spatial, vel_spatial) = model(
                h, loc.detach(), edges, vel.detach(), edge_attr
            )

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

        # loss = loss_mse(loc_spatial, loc_end)
        loss = loss_mse(loc_spatial, loc_end) + loss_mse(vel_spatial, vel_end)
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
                edges = loader.dataset.get_edges(B, N, L)
                edges = [edges[0].to(device), edges[1].to(device)]

                # generate features (vector norm) with dim (B, N, d) where d=1
                h_spatiotemporal_end = torch.norm(vel_end, p=2, dim=-1).unsqueeze(-1)

                # loc, vel are now (B * N * L, n)
                loc = loc.reshape(B * N * L, n)
                vel = vel.reshape(B * N * L, n)

                # h (or nodes) = ||v|| which is (B * N * L, d)
                h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
                rows, cols = edges

                # (B * N * (N - 1) * L, 1)
                loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
                    -1
                )  # relative distances among locations

                # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
                # where N(N - 1) is the num of connections in the graph
                edge_attr = edge_attr.view(B * N * (N - 1), 1)
                # (B * N * (N - 1), 1) -> (B * N * (N - 1), 1, 1)
                edge_attr = edge_attr.unsqueeze(1)
                # (B * N * (N - 1), 1, 1) -> (B * N * (N - 1), L, 1)
                edge_attr = edge_attr.expand(-1, L, -1)
                # (B * N * (N - 1), L, 1) -> (B * N * (N - 1) * L, 1)
                edge_attr = edge_attr.reshape(B * N * (N - 1) * L, 1)

                # (B * N * (N - 1) * L, 2)
                edge_attr = torch.cat(
                    [edge_attr, loc_dist], -1
                ).detach()  # concatenate all edge properties

                if args.time_exp:
                    torch.cuda.synchronize()
                    t1 = time.time()

                # EGNN_vel Baseline
                # h_spatial is (B, N, d)
                # loc_spatial is (B, N, n)
                # vel_spatial is (B, N, n)
                edges[0], edges[1] = edges[0].detach(), edges[1].detach()
                (h_spatial, loc_spatial, vel_spatial) = model(
                    h, loc.detach(), edges, vel.detach(), edge_attr
                )
            else:
                edges = loader.dataset.get_edges(B, N, 1)
                edges = [edges[0].to(device), edges[1].to(device)]

                # generate features (vector norm) with dim (B, N, d) where d=1
                h_spatial_end = torch.norm(vel_end, p=2, dim=-1).unsqueeze(-1)

                # loc, vel are (B, N, L, n)
                # take only the last entry at t=L
                loc = loc[:, :, -1, :]
                vel = vel[:, :, -1, :]

                # loc, vel are now (B * N, n)
                loc = loc.reshape(B * N, n)
                vel = vel.reshape(B * N, n)

                # h (or nodes) = ||v|| which is (B * N, d)
                h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
                rows, cols = edges

                # (B * N * (N - 1), 1)
                loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
                    -1
                )  # relative distances among locations

                # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
                # where N(N - 1) is the num of connections in the graph
                edge_attr = edge_attr.view(B * N * (N - 1), 1)

                # (B * N * (N - 1), 2)
                edge_attr = torch.cat(
                    [edge_attr, loc_dist], -1
                ).detach()  # concatenate all edge properties

                if args.time_exp:
                    torch.cuda.synchronize()
                    t1 = time.time()

                # EGNN_vel baseline
                # h_spatial is (B, N, d)
                # loc_spatial is (B, N, n)
                # vel_spatial is (B, N, n)
                edges[0], edges[1] = edges[0].detach(), edges[1].detach()
                (h_spatial, loc_spatial, vel_spatial) = model(
                    h, loc.detach(), edges, vel.detach(), edge_attr
                )

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

            # loss = loss_mse(loc_spatial, loc_end)
            loss = loss_mse(loc_spatial, loc_end) + loss_mse(vel_spatial, vel_end)
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
        # best_val_loss, best_test_loss, best_epoch = main()
        best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params = (
            main()
        )
        # Record best metrics
        text_out = f"Total Num Params: {total_params}\n"
        text_out += f"Best Train Loss Value: {best_train_loss}\n"
        text_out += f"Best Val Loss Value: {best_val_loss}\n"
        text_out += f"Best Test Loss Value: {best_test_loss}\n"

        file_name = (
            f"egnn_vel_baseline_timeseries_{args.time_series}_{args.dataset}.txt"
        )
        with open(file_name, "w") as file:
            file.write(text_out)
