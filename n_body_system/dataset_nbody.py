import numpy as np
import torch
import random

random.seed(0)
np.random.seed(0)


class NBodyDataset:
    """
    NBodyDataset

    """

    def __init__(
        self,
        partition="train",
        seq_len=100,
        horizon_len=1000,
        max_samples=1e8,
        dataset_name="se3_transformer",
        sample_freq=10,
    ):
        self.partition = partition
        if self.partition == "val":
            self.sufix = "valid"
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_10":
            self.sufix += "_charged10_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        elif dataset_name == "nbody_full":
            self.sufix += "_charged5_initvel1full"
        elif dataset_name == "nbody_long":
            self.sufix += "_charged5_initvel1long"
        elif dataset_name == "nbody_delta1_long":
            self.sufix += "_charged5_initvel1delta1_long"
        elif dataset_name == "nbody_5atoms":
            self.sufix += "_charged5_initvel15atoms"
        elif dataset_name == "nbody_20atoms":
            self.sufix += "_charged20_initvel120atoms"
        elif dataset_name == "nbody_30atoms":
            self.sufix += "_charged30_initvel130atoms"
        elif dataset_name == "nbody_100atoms":
            self.sufix += "_charged100_initvel1100atoms"
        elif dataset_name == "nbody_noisy":
            self.sufix += "_charged5_initvel1noisy"
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
        loc = np.load("n_body_system/dataset/loc_" + self.sufix + ".npy")
        vel = np.load("n_body_system/dataset/vel_" + self.sufix + ".npy")
        edges = np.load("n_body_system/dataset/edges_" + self.sufix + ".npy")
        charges = np.load("n_body_system/dataset/charges_" + self.sufix + ".npy")

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
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
            torch.Tensor(charges),
        )

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        # select at sample_freq
        # loc_seq = loc[: self.seq_len * self.sample_freq : self.sample_freq]
        # loc_seq = torch.transpose(loc_seq, 0, 1)

        # vel_seq = vel[: self.seq_len * self.sample_freq : self.sample_freq]
        # vel_seq = torch.transpose(vel_seq, 0, 1)

        # loc_end = loc[self.seq_len * self.sample_freq + self.horizon_len - 1]
        # vel_end = vel[self.seq_len * self.sample_freq + self.horizon_len - 1]
        loc_seq = loc[:-1]
        # loc_seq = loc[:10]
        loc_seq = torch.transpose(loc_seq, 0, 1)
        loc_end = loc[-1]

        vel_seq = vel[:-1]
        # vel_seq = vel[:10]
        vel_end = vel[-1]
        vel_seq = torch.transpose(vel_seq, 0, 1)

        return loc_seq, vel_seq, edge_attr, charges, loc_end, vel_end

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


if __name__ == "__main__":
    NBodyDataset(partition="train", dataset_name="nbody_small")
