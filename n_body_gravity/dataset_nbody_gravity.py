import numpy as np
import torch
import random

random.seed(0)
np.random.seed(0)


class NBodyGravityDataset:
    """
    NBodyDataset

    """

    def __init__(
        self,
        partition="train",
        seq_len=20,
        horizon_len=1000,
        max_samples=1e8,
        dataset_name="se3_transformer",
        sample_freq=20,
    ):
        self.partition = partition
        if self.partition == "val":
            self.sufix = "valid"
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody_gravity_50":
            self.sufix += "_gravity50_seqlen_10"
        elif dataset_name == "nbody_gravity_5":
            self.sufix += "_gravity5"
        elif dataset_name == "nbody_gravity_5_seqlen_50":
            self.sufix += "_gravity5_seqlen_50"
        elif dataset_name == "nbody_gravity_5_seqlen_100":
            self.sufix += "_gravity5_seqlen_100"
        elif dataset_name == "nbody_gravity_50_seqlen_50":
            self.sufix += "_gravity50_seqlen_50"
        elif dataset_name == "nbody_gravity_50_seqlen_100":
            self.sufix += "_gravity50_seqlen_100"
        elif dataset_name == "nbody_gravity_20":
            self.sufix += "_gravity20"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        print(f"Using Dataset: {dataset_name}")

        # L, H params
        self.seq_len = seq_len
        self.horizon_len = horizon_len
        self.sample_freq = sample_freq

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        loc = np.load("n_body_gravity/dataset/loc_" + self.sufix + ".npy")
        vel = np.load("n_body_gravity/dataset/vel_" + self.sufix + ".npy")
        edges = np.load("n_body_gravity/dataset/edges_" + self.sufix + ".npy")
        masses = np.load("n_body_gravity/dataset/masses_" + self.sufix + ".npy")

        loc, vel, edge_attr, edges, masses = self.preprocess(loc, vel, edges, masses)
        return (loc, vel, edge_attr, masses), edges

    def preprocess(self, loc, vel, edges, masses):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        loc, vel = torch.Tensor(loc).transpose(1, 2), torch.Tensor(vel).transpose(1, 2)
        n_nodes = loc.size(1)
        # TODO: max_samples -> NOT Sure whether to turn on/off
        # loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        # vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        # charges = charges[0 : self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = (
            torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)
        )  # swap n_nodes <--> batch_size and add nf dimension

        # Only extract relevant values
        # loc = torch.cat((loc[:, : self.seq_len * self.sample_freq : self.sample_freq, :, :], loc[:, self.seq_len * self.sample_freq + self.horizon_len - 1, :, :].unsqueeze(1)), dim=1)
        # vel = torch.cat((vel[:, : self.seq_len * self.sample_freq : self.sample_freq, :, :], vel[:, self.seq_len * self.sample_freq + self.horizon_len - 1, :, :].unsqueeze(1)), dim=1)

        return (
            torch.Tensor(loc),
            torch.Tensor(vel),
            torch.Tensor(edge_attr),
            edges,
            torch.Tensor(masses),
        )

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, masses = self.data
        loc, vel, edge_attr, masses = loc[i], vel[i], edge_attr[i], masses[i]

        # select at sample_freq
        # loc_seq = loc[: self.seq_len * self.sample_freq : self.sample_freq]
        # loc_seq = torch.transpose(loc_seq, 0, 1)

        # vel_seq = vel[: self.seq_len * self.sample_freq : self.sample_freq]
        # vel_seq = torch.transpose(vel_seq, 0, 1)

        # loc_end = loc[self.seq_len * self.sample_freq + self.horizon_len - 1]
        # vel_end = vel[self.seq_len * self.sample_freq + self.horizon_len - 1]
        loc_seq = loc[:, :-1, :]
        loc_end = loc[:, -1, :]

        vel_seq = vel[:, :-1, :]
        vel_end = vel[:, -1, :]

        # loc_seq = loc[:-1]
        # # loc_seq = loc[:10]
        # loc_seq = torch.transpose(loc_seq, 0, 1)
        # loc_end = loc[-1]

        # vel_seq = vel[:-1]
        # # vel_seq = vel[:10]
        # vel_end = vel[-1]
        # vel_seq = torch.transpose(vel_seq, 0, 1)

        N, L, _ = loc_seq.size()

        # # Create the base sequence 0 to 9
        # base_sequence = torch.arange(10)  # [0, 1, 2, ..., 9]

        # # Adjust the sequence based on i, so we get [0, 1, 2, ..., 9] for i=0 and [10, 11, 12, ..., 19] for i=1
        # adjusted_sequence = 10 * i + base_sequence  # [10*i + 0, 10*i + 1, ..., 10*i + 9]

        # # Repeat each number 12 times to create sequences like [0, 0, ..., 0, 1, 1, ..., 1, ..., 9, 9, ..., 9]
        # batch = adjusted_sequence.repeat_interleave(12)

        # Create the base sequence 0 to L - 1
        base_sequence = torch.arange(L)  # [0, 1, 2, ..., 9]

        # Adjust the sequence based on i, so we get [0, 1, 2, ..., 9] for i=0 and [10, 11, 12, ..., 19] for i=1
        adjusted_sequence = L * i + base_sequence  # [10*i + 0, 10*i + 1, ..., 10*i + 9]

        # Repeat each number 20 (number of masses N) times to create sequences like [0, 0, ..., 0, 1, 1, ..., 1, ..., 9, 9, ..., 9]
        batch = adjusted_sequence.repeat_interleave(N)

        # print(f"batch: {batch.shape}")
        # breakpoint()

        return loc_seq, vel_seq, edge_attr, masses, loc_end, vel_end, batch

    def __len__(self):
        return len(self.data[0])

    # def get_edges(self, batch_size, n_nodes):
    #     edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
    #     if batch_size == 1:
    #         return edges
    #     elif batch_size > 1:
    #         rows, cols = [], []
    #         for i in range(batch_size):
    #             rows.append(edges[0] + n_nodes * i)
    #             cols.append(edges[1] + n_nodes * i)
    #         edges = [torch.cat(rows), torch.cat(cols)]
    #     return edges
    def get_edges(self, batch_size, n_nodes, seq_len):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        rows, cols = [], []
        for i in range(batch_size * seq_len):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]

        return edges


def custom_collate(data_batch):
    loc_seq, vel_seq, edge_attr, masses, loc_end, vel_end, batch = data_batch[0]

    # loc = torch.cat([data[0] for data in data_batch], dim=0)  # Shape: (B_total, 3)
    loc_seq = torch.stack([data[0] for data in data_batch], dim=0)
    vel_seq = torch.stack([data[1] for data in data_batch], dim=0)
    edge_attr = torch.stack([data[2] for data in data_batch], dim=0)
    masses = torch.stack([data[3] for data in data_batch], dim=0)
    loc_end = torch.stack([data[4] for data in data_batch], dim=0)
    vel_end = torch.stack([data[5] for data in data_batch], dim=0)
    batch = torch.stack([data[6] for data in data_batch], dim=0)
    B = batch.size(0)
    L = loc_seq.size(-2)
    N = loc_seq.size(1)

    batch_idx = torch.arange(B * L).repeat_interleave(N)

    return loc_seq, vel_seq, edge_attr, masses, loc_end, vel_end, batch_idx


if __name__ == "__main__":
    NBodyGravityDataset(partition="train", dataset_name="nbody_gravity_50")
