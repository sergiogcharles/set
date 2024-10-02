import argparse
import torch
from n_body_gravity.dataset_nbody_gravity import NBodyGravityDataset
from torch.utils.tensorboard import SummaryWriter

import optuna
from optuna.trial import TrialState
import datetime
from tqdm import tqdm

from n_body_gravity.model import SE3TransformerBaseline
import os
from torch import nn, optim
import json
import time
from torch.autograd import Variable

torch.manual_seed(0)

parser = argparse.ArgumentParser(
    description="SE(3)-Transformer Baseline Trajectory Prediction Example"
)


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


parser.add_argument(
    "--exp_name",
    type=str,
    default="se3_transformer_baseline",
    help="experiment_name",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="input batch size for training (default: 128)",
)

parser.add_argument(
    "--seq_len",
    type=int,
    default=10,
    help="sequence length during training.",
)

parser.add_argument(
    "--horizon_len",
    type=int,
    default=1000,
    help="horizon length during training.",
)

parser.add_argument(
    "--sample_freq",
    type=int,
    default=10,
    help="How frequently to sample sequence data for model.",
)

parser.add_argument(
    "--num_atoms",
    type=int,
    default=20,
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

# parser.add_argument(
#     "--hidden_nf",
#     type=int,
#     default=64,
#     help="hidden dimension",
# )
parser.add_argument(
    "--hidden_nf",
    type=int,
    default=16,
    help="hidden dimension",
)

parser.add_argument(
    "--proj_nf",
    type=int,
    default=3,
    help="projection dimension",
)

parser.add_argument(
    "--heads",
    type=int,
    default=1,
    help="number of heads",
)

parser.add_argument(
    "--depth",
    type=int,
    default=2,
    help="depth",
)

parser.add_argument(
    "--dim_head",
    type=int,
    default=16,
    help="dimension of head",
)

parser.add_argument(
    "--num_degrees",
    type=int,
    default=2,
    help="number of degrees",
)

parser.add_argument(
    "--valid_radius",
    type=int,
    default=4,
    help="valid radius",
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
    "--dataset",
    type=str,
    default="nbody_gravity_20",
    metavar="N",
    help="nbody_small, nbody, nbody_",
)
parser.add_argument("--time_exp", type=int, default=0, help="timing experiment")

parser.add_argument("--norm_diff", type=eval, default=False, help="normalize_diff")

parser.add_argument("--recurrent", type=eval, default=True, help="use recurrent")

parser.add_argument(
    "--attention", type=str_to_bool, default=True, help="use edge inference"
)

parser.add_argument("--tanh", type=eval, default=False, help="use tanh")

parser.add_argument(
    "--tensorboard", type=str_to_bool, default=False, help="Uses tensorboard"
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
        self, patience=1, min_delta=0, plateau_patience=5, plateau_threshold=1e-4
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
            print(f"Counter: {self.counter}")
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
    batch_size,
    weight_decay,
    heads,
    depth,
    dim_head,
    num_degrees,
    valid_radius,
    num_atoms,
    optuna=False,
):
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./runs/{dt}_se3_transformer_baseline_optuna_{optuna}_lr_{lr}_bsz_{batch_size}_wd_{weight_decay}_heads_{heads}_depth_{depth}_dhead_{dim_head}_ndegrees_{num_degrees}_rad_{valid_radius}_natoms_{num_atoms}/"

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
    # seq_len = args.seq_len
    # num_atoms = args.num_atoms
    # in_node_nf = args.in_node_nf
    # in_edge_nf = args.in_edge_nf
    # in_pos_nf = args.in_pos_nf
    # hidden_nf = args.hidden_nf
    # proj_nf = args.proj_nf
    # spatial_attention = args.spatial_attention
    # temporal_attention = args.temporal_attention
    # positional_encoding = args.positional_encoding
    # causal_attention = args.causal_attention
    # dropout = args.dropout
    # kappa = args.kappa
    # equivariant = args.equivariant

    # n_egnn_layers = args.n_egnn_layers
    # n_stacking_layers = args.n_stacking_layers

    # recurrent = args.recurrent
    # norm_diff = args.norm_diff
    # tanh = args.tanh
    # adjacency = args.adjacency

    # Hyperparameters for Optuna search
    # lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [32, 100, 256])
    # weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-6])
    # n_egnn_layers = trial.suggest_categorical("n_egnn_layers", [2, 3])
    # n_stacking_layers = trial.suggest_categorical("n_stacking_layers", [2, 3])
    # hidden_nf = trial.suggest_categorical("hidden_nf", [128, 256, 512])
    # spatial_attention = trial.suggest_categorical(
    #     "spatial_attention", [args.spatial_attention]
    # )
    # temporal_attention = trial.suggest_categorical(
    #     "temporal_attention", [args.temporal_attention]
    # )
    # positional_encoding = trial.suggest_categorical(
    #     "positional_encoding", [True, False]
    # )
    # causal_attention = trial.suggest_categorical("causal_attention", [True, False])
    # dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.5])
    # kappa = trial.suggest_categorical("kappa", [1e4, 1e5])
    # equivariant = trial.suggest_categorical("equivariant", [args.equivariant])
    # adjacency = trial.suggest_categorical("adjacency", [args.adjacency])
    # recurrent = trial.suggest_categorical("recurrent", [True, False])

    batch_size = trial.suggest_categorical("batch_size", [32, 100, 256])
    seq_len = args.seq_len
    num_atoms = args.num_atoms
    in_node_nf = args.in_node_nf
    heads = args.heads
    depth = args.depth
    dim_head = args.dim_head
    num_degrees = args.num_degrees
    valid_radius = args.valid_radius

    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-6])
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    proj_nf = args.proj_nf

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Dataset
    dataset_train = NBodyGravityDataset(
        partition="train",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
        sample_freq=args.sample_freq,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyGravityDataset(
        partition="val",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, drop_last=True
    )

    dataset_test = NBodyGravityDataset(
        partition="test",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, drop_last=True
    )

    model = SE3TransformerBaseline(
        dim=in_node_nf,
        heads=heads,
        depth=depth,
        dim_head=dim_head,
        num_degrees=num_degrees,
        valid_radius=valid_radius,
        proj_nf=proj_nf,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(
            lr,
            batch_size,
            weight_decay,
            heads,
            depth,
            dim_head,
            num_degrees,
            valid_radius,
            num_atoms,
            optuna=True,
        )

    # early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.05)

    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # if epoch % args.test_interval == 0:
        val_loss = val(
            model,
            epoch,
            loader_val,
        )
        test_loss = val(
            model,
            epoch,
            loader_test,
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
            torch.save(
                model.state_dict(),
                f"best_models/se3_transformer_baseline_optuna_{True}_lr_{lr}_bsz_{batch_size}_wd_{weight_decay}_heads_{heads}_depth_{depth}_dhead_{dim_head}_ndegrees_{num_degrees}_radius_{valid_radius}_{num_atoms}.pt",
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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        direction="minimize",
    )
    study.optimize(objective, n_trials=30)

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
    file_name = f"optuna_study_se3_transformer_baseline_equivariant_{trial.params['equivariant']}.txt"
    with open(file_name, "w") as file:
        file.write(text_out)


def main():
    dataset_train = NBodyGravityDataset(
        partition="train",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        max_samples=args.max_training_samples,
        sample_freq=args.sample_freq,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    dataset_val = NBodyGravityDataset(
        partition="val",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    dataset_test = NBodyGravityDataset(
        partition="test",
        seq_len=args.seq_len,
        horizon_len=args.horizon_len,
        dataset_name=args.dataset,
        sample_freq=args.sample_freq,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=True
    )

    # Hyperparameters
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_atoms = args.num_atoms
    in_node_nf = args.in_node_nf
    heads = args.heads
    depth = args.depth
    dim_head = args.dim_head
    num_degrees = args.num_degrees
    valid_radius = args.valid_radius

    weight_decay = args.weight_decay
    lr = args.lr
    proj_nf = args.proj_nf

    model = SE3TransformerBaseline(
        dim=in_node_nf,
        heads=heads,
        depth=depth,
        dim_head=dim_head,
        num_degrees=num_degrees,
        valid_radius=valid_radius,
        proj_nf=proj_nf,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_params}")

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.tensorboard:
        writer = create_summary_writer(
            lr,
            batch_size,
            weight_decay,
            heads,
            depth,
            dim_head,
            num_degrees,
            valid_radius,
            num_atoms,
            optuna=False,
        )

    # early stopping
    early_stopper = EarlyStopper(patience=10, min_delta=0.005)
    results = {"epochs": [], "losess": []}
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        if args.tensorboard:
            writer.add_scalar("Loss/train", train_loss, epoch)

        # if epoch % args.test_interval == 0:
        val_loss = val(
            model,
            epoch,
            loader_val,
        )
        test_loss = val(
            model,
            epoch,
            loader_test,
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
                f"best_models/se3_transformer_baseline_optuna_{False}_lr_{lr}_bsz_{batch_size}_wd_{weight_decay}_heads_{heads}_depth_{depth}_dhead_{dim_head}_ndegrees_{num_degrees}_radius_{valid_radius}_{num_atoms}.pt",
            )

        print(
            "*** Best Train Loss: %.5f \t Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d"
            % (best_train_loss, best_val_loss, best_test_loss, best_epoch)
        )

        # json_object = json.dumps(results, indent=4)
        # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
        #     outfile.write(json_object)
        if early_stopper.early_stop(val_loss):
            print(f"EARLY STOPPED")
            break

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params


def train(model, optimizer, epoch, loader):
    model.train()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    for data in tqdm(loader):
        # B = batch_size, N = n_nodes, L = seq_len, n = 3
        B, N, L, n = data[0].size()
        data = [d.to(device) for d in data]
        # loc, vel has dim (B, N, L, n) where n=3
        # loc_end, vel_end has dim (B, N, n)
        loc, vel, edge_attr, masses, loc_end, vel_end, batch = data

        # print(
        #     f"loc: {loc.shape}, vel: {vel.shape}, loc_end {loc_end.shape}, vel_end {vel_end.shape} {edge_attr.shape}"
        # )
        # breakpoint()

        edges = loader.dataset.get_edges(B, N, L)
        edges = [edges[0].to(device), edges[1].to(device)]

        # loc, vel are now (B * N * L, n)
        # loc = loc.reshape(B * N * L, n)
        loc = loc.reshape(B, N, L, n)[:, :, 0, :]
        vel = vel.reshape(B, N, L, n)[:, :, 0, :]

        # h (or nodes) = ||v|| which is (B * N * L, d)
        h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
        # masses = masses.unsqueeze(2).expand(-1, -1, L, -1).reshape(-1, 1)
        # h = torch.cat((h, masses), dim=-1)

        # rows, cols = edges

        # # (B * N * (N - 1) * L, 1)
        # loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
        #     -1
        # )  # relative distances among locations

        # # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
        # # where N(N - 1) is the num of connections in the graph
        # edge_attr = edge_attr.view(B * N * (N - 1), 1)
        # # (B * N * (N - 1), 1) -> (B * N * (N - 1), 1, 1)
        # edge_attr = edge_attr.unsqueeze(1)
        # # (B * N * (N - 1), 1, 1) -> (B * N * (N - 1), L, 1)
        # edge_attr = edge_attr.expand(-1, L, -1)
        # # (B * N * (N - 1), L, 1) -> (B * N * (N - 1) * L, 1)
        # edge_attr = edge_attr.reshape(B * N * (N - 1) * L, 1)

        # # (B * N * (N - 1) * L, 2)
        # edge_attr = torch.cat(
        #     [edge_attr, loc_dist], -1
        # ).detach()  # concatenate all edge properties

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        # Spacetime Transformer
        # h_spatiotemporal is (B, N, d)
        # loc_spatiotemporal is (B, N, n)
        # vel_spatiotemporal is (B, N, n)
        edges[0], edges[1] = edges[0].detach(), edges[1].detach()
        # (h_spatiotemporal, loc_spatiotemporal, vel_spatiotemporal) = model(
        #     h, loc.detach(), edges, vel.detach(), edge_attr
        # )
        loc_spatiotemporal = model(h, loc.detach())

        # print(f"loc: {loc_spatiotemporal.shape} {loc_end.shape}")
        # breakpoint()

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

        loss = loss_mse(loc_spatiotemporal, loc_end)

        # loss = loss_mse(loc_spatiotemporal, loc_end) + loss_mse(
        #     vel_spatiotemporal, vel_end
        # )
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


def val(model, epoch, loader):
    model.eval()

    res = {"epoch": epoch, "loss": 0, "coord_reg": 0, "counter": 0}

    with torch.no_grad():
        for data in tqdm(loader):
            # B = batch_size, N = n_nodes, L = seq_len, n = 3
            B, N, L, n = data[0].size()
            data = [d.to(device) for d in data]
            # loc, vel has dim (B, N, L, n) where n=3
            # loc_end, vel_end has dim (B, N, n)
            loc, vel, edge_attr, masses, loc_end, vel_end, batch = data

            # print(
            #     f"loc: {loc.shape}, vel: {vel.shape}, loc_end {loc_end.shape}, vel_end {vel_end.shape}"
            # )

            edges = loader.dataset.get_edges(B, N, L)
            edges = [edges[0].to(device), edges[1].to(device)]

            # loc, vel are now (B * N * L, n)
            # loc = loc.reshape(B * N * L, n)
            # vel = vel.reshape(B * N * L, n)
            loc = loc.reshape(B, N, L, n)[:, :, 0, :]
            vel = vel.reshape(B, N, L, n)[:, :, 0, :]

            # h (or nodes) = ||v|| which is (B * N * L, d)
            h = torch.sqrt(torch.sum(vel**2, dim=-1)).unsqueeze(-1).detach()
            # masses = masses.unsqueeze(2).expand(-1, -1, L, -1).reshape(-1, 1)
            # h = torch.cat((h, masses), dim=-1)

            rows, cols = edges

            # (B * N * (N - 1) * L, 1)
            # loc_dist = torch.sum((loc[rows, :] - loc[cols, :]) ** 2, -1).unsqueeze(
            #     -1
            # )  # relative distances among locations

            # # (B, N * (N-1), 1) -> (B * N * (N - 1), 1)
            # # where N(N - 1) is the num of connections in the graph
            # edge_attr = edge_attr.view(B * N * (N - 1), 1)
            # # (B * N * (N - 1), 1) -> (B * N * (N - 1), 1, 1)
            # edge_attr = edge_attr.unsqueeze(1)
            # # (B * N * (N - 1), 1, 1) -> (B * N * (N - 1), L, 1)
            # edge_attr = edge_attr.expand(-1, L, -1)
            # # (B * N * (N - 1), L, 1) -> (B * N * (N - 1) * L, 1)
            # edge_attr = edge_attr.reshape(B * N * (N - 1) * L, 1)

            # # (B * N * (N - 1) * L, 2)
            # edge_attr = torch.cat(
            #     [edge_attr, loc_dist], -1
            # ).detach()  # concatenate all edge properties

            if args.time_exp:
                torch.cuda.synchronize()
                t1 = time.time()

            # Spacetime Transformer
            # h_spatiotemporal is (B, N, d)
            # loc_spatiotemporal is (B, N, n)
            # vel_spatiotemporal is (B, N, n)
            edges[0], edges[1] = edges[0].detach(), edges[1].detach()
            # (h_spatiotemporal, loc_spatiotemporal, vel_spatiotemporal) = model(
            #     h, loc.detach(), edges, vel.detach(), edge_attr
            # )
            loc_spatiotemporal = model(h, loc.detach())

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

            loss = loss_mse(loc_spatiotemporal, loc_end)

            # loss = loss_mse(loc_spatiotemporal, loc_end) + loss_mse(
            #     vel_spatiotemporal, vel_end
            # )
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
        best_train_loss, best_val_loss, best_test_loss, best_epoch, total_params = (
            main()
        )
        # Record best metrics
        text_out = f"Total Num Params: {total_params}\n"
        text_out += f"Best Train Loss Value: {best_train_loss}\n"
        text_out += f"Best Val Loss Value: {best_val_loss}\n"
        text_out += f"Best Test Loss Value: {best_test_loss}\n"

        file_name = f"se3_transformer_baseline_equivariant_lr_{args.lr}_bsz_{args.batch_size}_wd_{args.weight_decay}_heads_{args.heads}_depth_{args.depth}_dhead_{args.dim_head}_ndegrees_{args.num_degrees}_radius_{args.valid_radius}_{args.num_atoms}_dataset_{args.dataset}.txt"
        with open(file_name, "w") as file:
            file.write(text_out)
