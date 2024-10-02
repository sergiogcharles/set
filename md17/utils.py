import torch


def compute_mean_std_timeseries(dataloader, label_property):
    dataset = dataloader.dataset
    values = []

    # Check if the dataset is a Subset
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset  # Access the original dataset
        indices = dataset.indices  # Get the indices of the subset

        for idx in indices:
            data = original_dataset[idx]  # Get the data object
            values.append(getattr(data, label_property))

    else:
        for data in dataset:
            values.append(getattr(data, label_property))

    # Stack all collected values into a single tensor
    values = torch.cat(values, dim=0)

    mean = torch.mean(values)
    std = torch.std(values)

    print(f"mean: {mean}, std: {std}")

    return mean, std


def compute_mean_std(dataloader):
    positions = []

    for data in dataloader:
        # Extract energy from the batch and add to the list
        # print(f"data: {data.pos.shape}")
        # breakpoint()
        positions.append(data.pos)

    # Concatenate all position values into a single tensor
    positions = torch.cat(positions, dim=0)

    # Compute mean and standard deviation
    mean = positions.mean(dim=0)
    std = positions.std(dim=0)

    print(f"mean: {mean}, std: {std}")

    return mean, std


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


# def get_edges_timeseries(batch_size, n_nodes, seq_len, device):
#     # edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
#     rows, cols = [], []
#     # for i in range(batch_size * seq_len):
#     #     rows.append(edges[0] + n_nodes * i)
#     #     cols.append(edges[1] + n_nodes * i)
#     for batch_idx in range(batch_size * seq_len):
#         for i in range(n_nodes):
#             for j in range(n_nodes):
#                 rows.append(i + batch_idx * n_nodes)
#                 cols.append(j + batch_idx * n_nodes)
#     edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]

#     return edges


# def get_edges(batch_size, n_nodes, device):
#     # edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
#     rows, cols = [], []
#     # for i in range(batch_size):
#     #     rows.append(edges[0] + n_nodes * i)
#     #     cols.append(edges[1] + n_nodes * i)
#     # edges = [torch.cat(rows), torch.cat(cols)]
#     for batch_idx in range(batch_size):
#         for i in range(n_nodes):
#             for j in range(n_nodes):
#                 rows.append(i + batch_idx * n_nodes)
#                 cols.append(j + batch_idx * n_nodes)

#     edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]

#     return edges
