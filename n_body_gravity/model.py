import torch
from torch import nn, Tensor
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel
from torch.nn import functional as F
import math

from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.io import fs
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor


class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver("sum" if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer("atomic_mass", atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer("initial_atomref", atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # print(f"h before: {z.shape}")

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)

        # print(f"edge index: {edge_index.shape} {edge_index[0][:100]} {h.shape}")
        # breakpoint()

        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        # out = self.readout(h, batch, dim=0)
        # print(f"h: {h.shape} out {out.shape}")
        # breakpoint()

        # if self.dipole:
        #     out = torch.norm(out, dim=-1, keepdim=True)

        # if self.scale is not None:
        #     out = self.scale * out

        # return out
        if self.dipole:
            h = torch.norm(h, dim=-1, keepdim=True)

        if self.scale is not None:
            h = self.scale * h

        return h

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        # print(f"batch: {batch.shape} {pos.shape} {batch[:100]}")
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        # print(f"pos: {pos.shape} {edge_index.shape} {edge_index[0].shape}")
        # print(edge_index[0][:100])
        # breakpoint()
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(
        self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float
    ):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        # print(
        #     f"edge_index: {edge_index.shape}, {edge_index[0][:100]} {edge_index[1][:100]}"
        # )
        # breakpoint()
        # print(f"x: {x.shape} {edge_index.max()}")
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        # print(f"edge index: {edge_index.shape} {x.shape}")
        # breakpoint()

        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        attention=0,
        recurrent=False,
    ):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        # self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_nf=1,
                    act_fn=act_fn,
                    attention=attention,
                    recurrent=recurrent,
                ),
            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 3)
        )
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        # h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        # return h
        return self.decoder(h)


def get_velocity_attr(loc, vel, rows, cols):
    # return  torch.cat([vel[rows], vel[cols]], dim=1)

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff / norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        proj_nf,
        device="cpu",
        act_fn=nn.LeakyReLU(0.2),
        n_layers=4,
        coords_weight=1.0,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.proj_nf = proj_nf
        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                ),
            )
        # projection layer
        self.projection = nn.Linear(self.hidden_nf, self.proj_nf)

        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        # print(f'edge_attr2: {edge_attr.shape}')
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            # if vel is not None:
            # vel_attr = get_velocity_attr(x, vel, edges[0], edges[1])
            # edge_attr = torch.cat([edge_attr0, vel_attr], dim=1).detach()
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)

        h = self.projection(h)

        return h, x, vel


class EGNN_vel(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        proj_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=True,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super(EGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.proj_nf = proj_nf
        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL_vel(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    attention=attention,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.projection = nn.Linear(self.hidden_nf, self.proj_nf)

        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        # print(f'edge_attr egnn vel: {edge_attr.shape}')
        # print(f'h: {h.shape}')
        # breakpoint()
        h = self.embedding(h)
        # print(f'h: {h.shape}')
        # breakpoint()
        for i in range(0, self.n_layers):
            h, x, vel, _ = self._modules["gcl_%d" % i](
                h, edges, x, vel, edge_attr=edge_attr
            )
        # project down
        h = self.projection(h)

        return h, x, vel


class EGNNvelSchnet(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        proj_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        attention=True,
        norm_diff=False,
        tanh=False,
        hidden_channels=32,
        num_filters=32,
        num_interactions=6,
        cutoff=10.0,
    ):
        super(EGNNvelSchnet, self).__init__()

        self.egnn_vel = EGNN_vel(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            proj_nf=proj_nf,
            device=device,
            act_fn=act_fn,
            n_layers=n_layers,
            coords_weight=coords_weight,
            attention=attention,
            recurrent=recurrent,
            norm_diff=norm_diff,
            tanh=tanh,
        )

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            cutoff=cutoff,
        )

    def forward(self, h, x, edges, vel, edge_attr, batch):
        h = h.int()
        h = self.schnet(h, x, batch)
        # print(f'egnn vel schnetnet edge attr: {edge_attr.shape}')
        # breakpoint()
        h, x, vel = self.egnn_vel(h, x, edges, vel, edge_attr)
        return h, x, vel


class EGNNSchnet(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        proj_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        hidden_channels=32,
        num_filters=32,
        num_interactions=6,
        cutoff=10.0,
    ):
        super(EGNNSchnet, self).__init__()
        self.egnn = EGNN(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            proj_nf,
            device,
            act_fn,
            n_layers,
            coords_weight,
        )

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            cutoff=cutoff,
        )

    def forward(self, h, x, edges, vel, edge_attr, batch):
        h = h.squeeze()
        h = self.schnet(h, x, batch)
        h, x = self.egnn(h, x, edges, edge_attr)
        return h, x, vel
        # return h, x


class PositionalEncoder(nn.Module):

    def __init__(
        self,
        B,
        N,
        L,
        d,
        n,
        kappa=10000,
        dropout=0.1,
        device="cpu",
        use_velocity=True,
    ):
        super().__init__()
        self.B = B
        self.N = N
        self.L = L
        self.d = d
        self.n = n
        self.kappa = kappa
        self.device = device
        self.use_velocity = use_velocity

        self.dropout = nn.Dropout(p=dropout)

        # (L, d)
        self.pe_d = self.positional_encoder_matrix(self.d)
        # (L, d) -> (B, N, L, d)
        self.pe_d = self.pe_d.unsqueeze(0).unsqueeze(0)
        self.pe_d = self.pe_d.expand(self.B, self.N, self.L, self.d)

        # (L, n)
        self.pe_n = self.positional_encoder_matrix(self.n)
        # (L, n) -> (B, N, L, n)
        self.pe_n = self.pe_n.unsqueeze(0).unsqueeze(0)
        self.pe_n = self.pe_n.expand(self.B, self.N, self.L, self.n)

        # (L, N * (N - 1) * 2)
        self.pe_edge = self.positional_encoder_matrix(self.N * (self.N - 1) * 2)
        # (L, N * (N - 1) * 2) -> (B, L, N * (N - 1) * 2)
        self.pe_edge = self.pe_edge.unsqueeze(0)
        self.pe_edge = self.pe_edge.expand(self.B, self.L, self.N * (self.N - 1) * 2)

    def positional_encoder_matrix(self, dim):
        # (L, dim)
        pe = torch.zeros((self.L, dim)).to(self.device)
        for j in range(self.L):
            for i in range(dim):
                if i % 2 == 0:
                    pe[j, i] = math.sin(j / (self.kappa ** (i / dim)))
                else:
                    pe[j, i] = math.cos(j / (self.kappa ** ((i - 1) / dim)))
        return pe

    def forward(self, x):
        # theta -> (B, N, L, d)
        # xi -> (B, N, L, n)
        # omega -> (B, N, L, n)
        # edge_attr -> (B, L, N * (N - 1) * 2)
        if self.use_velocity:
            (theta, xi, omega, edge_attr) = x
        else:
            (theta, xi, edge_attr) = x

        theta = theta + self.pe_d
        theta = self.dropout(theta)

        xi = xi + self.pe_n
        xi = self.dropout(xi)

        if self.use_velocity:
            omega = omega + self.pe_n
            omega = self.dropout(omega)

        if not (edge_attr is None):
            edge_attr = edge_attr + self.pe_edge
            edge_attr = self.dropout(edge_attr)

        if self.use_velocity:
            return (theta, xi, omega, edge_attr)
        else:
            return (theta, xi, edge_attr)


class ETAL(nn.Module):
    # n = dim of position
    # d = dim of features
    def __init__(
        self,
        B,
        N,
        L,
        d,
        n,
        dropout=0.1,
        kappa=10000,
        positional_encoding=True,
        equivariant=True,
        causal_attention=True,
        device="cpu",
    ):
        super(ETAL, self).__init__()
        self.equivariant = equivariant
        self.device = device

        # key, query, value matrices for feature space
        self.K_features = nn.Linear(d, d, bias=False).to(self.device)
        self.Q_features = nn.Linear(d, d, bias=False).to(self.device)
        self.V_features = nn.Linear(d, d, bias=False).to(self.device)

        if not self.equivariant:
            self.K_loc = nn.Linear(n, n, bias=False).to(self.device)
            self.Q_loc = nn.Linear(n, n, bias=False).to(self.device)
            self.V_loc = nn.Linear(n, n, bias=False).to(self.device)

        # example scalar hyperparameter
        self.B_const = 0.5
        # whether to use causal attention mask
        self.causal_attention = causal_attention
        # whether to use positional encoding
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.positional_encoder = PositionalEncoder(
                B,
                N,
                L,
                d,
                n,
                kappa=kappa,
                dropout=dropout,
                device=device,
                use_velocity=False,
            )

    def compute_vectorized_transform(self, xi, B_const):
        B, N, L, n = xi.size()

        # Difference matrix D_i = (B, N, L, 1, n) - (B, N, 1, L, n)
        D = xi.unsqueeze(3) - xi.unsqueeze(2)  # Shape: (B, N, L, L, n)

        # Compute squared differences
        squared_diffs = torch.sum(D**2, dim=-1)  # Shape: (B, N, L, L)

        if self.causal_attention:
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(L, L), diagonal=1
            ).bool()  # Shape: (L, L)
            causal_mask = (
                causal_mask.unsqueeze(0).unsqueeze(0).to(xi.device)
            )  # Shape: (1, 1, L, L)

            # Apply mask to squared differences
            squared_diffs.masked_fill_(causal_mask, -float("inf"))

            # Compute beta_i using the softmax of the squared norm of the differences
            beta = F.softmax(squared_diffs, dim=-1)  # Shape: (B, N, L, L)
        else:
            # Compute beta_i using the softmax of the squared norm of the differences
            beta = F.softmax(squared_diffs, dim=-1)  # Shape: (B, N, L, L)

        # Define identity and ones matrices
        I = (
            torch.eye(L).unsqueeze(0).unsqueeze(0).to(self.device)
        )  # Shape: (1, 1, L, L)
        ones = torch.ones_like(I).to(self.device)  # Shape: (1, 1, L, L)

        # (1_{LxL} - I_{LxL}) element-wise multiplied by beta
        mask = ones - I
        masked_beta = mask * beta  # Shape: (B, N, L, L)

        # First term: xi
        xi_term = xi  # Shape: (B, N, L, n)

        # Check for term 2
        # calc1 = masked_beta[0, 0, :, :] @ xi_term[0, 0, :, :]
        # calc2 = torch.matmul(masked_beta, xi)[0, 0, :, :]
        # print(torch.equal(calc1, calc2))

        # Second term: B * (masked_beta @ xi)
        second_term = B_const * torch.matmul(masked_beta, xi)  # Shape: (B, N, L, n)

        # Third term: B * ((masked_beta @ ones) * xi)
        sum_masked_beta = torch.sum(
            masked_beta, dim=-1, keepdim=True
        )  # Shape: (B, N, L, 1)
        third_term = B_const * sum_masked_beta * xi  # Shape: (B, N, L, n)

        # Check for term 3
        # calc1 = xi[0, 0, :, :] * sum_masked_beta[0, 0, :, :]
        # calc2 = (sum_masked_beta * xi)[0, 0, :, :]
        # print(torch.equal(calc1, calc2))

        # Compute v
        v = xi_term + second_term - third_term  # Shape: (B, N, L, n)

        return v

    # theta corresponds to h (features), xi to x (positions), and omega to v (velocities)
    # theta has dim (B, N, L, d)
    # xi has dim (B, N, L, n)
    # omega has dim (B, N, L, n)
    # edge_attr has dim (B, L, N * (N-1) * 2)
    # B = batch size
    # N = num. atoms
    # L = seq length
    # n = dim of space, vel
    # d = feature space dim
    def forward(self, input):
        (theta, xi, edge_attr) = input
        # theta = Variable(theta.data, requires_grad=True)
        # xi = Variable(xi.data, requires_grad=True)
        # omega = Variable(omega.data, requires_grad=True)

        # Positional encoder
        if self.positional_encoding:
            (theta, xi, edge_attr) = self.positional_encoder((theta, xi, edge_attr))

        # 1. Feature space
        key_features = self.K_features(theta)
        query_features = self.Q_features(theta)
        value_features = self.V_features(theta)

        h_out = F.scaled_dot_product_attention(
            query_features,
            key_features,
            value_features,
            is_causal=self.causal_attention,
        )

        # 2. Position space, equivariant to E(n)
        if self.equivariant:
            x_out = self.compute_vectorized_transform(xi, self.B_const)
        else:
            key_loc = self.K_loc(xi)
            query_loc = self.Q_loc(xi)
            value_loc = self.V_loc(xi)

            x_out = F.scaled_dot_product_attention(
                query_loc, key_loc, value_loc, is_causal=self.causal_attention
            )

        return (h_out, x_out, edge_attr)


class ETAL_vel(nn.Module):
    # n = dim of position, velocity
    # d = dim of features
    def __init__(
        self,
        B,
        N,
        L,
        d,
        n,
        dropout=0.1,
        kappa=10000,
        adjacency=True,
        positional_encoding=True,
        equivariant=True,
        causal_attention=True,
        device="cpu",
    ):
        super(ETAL_vel, self).__init__()
        self.equivariant = equivariant
        # whether we perform self-attention on adjacency matrix
        self.adjacency = adjacency

        self.device = device

        # key, query, value matrices for feature space
        self.K_features = nn.Linear(d, d, bias=False).to(self.device)
        self.Q_features = nn.Linear(d, d, bias=False).to(self.device)
        self.V_features = nn.Linear(d, d, bias=False).to(self.device)

        if not self.equivariant:
            self.K_loc = nn.Linear(n, n, bias=False).to(self.device)
            self.Q_loc = nn.Linear(n, n, bias=False).to(self.device)
            self.V_loc = nn.Linear(n, n, bias=False).to(self.device)

            self.K_vel = nn.Linear(n, n, bias=False).to(self.device)
            self.Q_vel = nn.Linear(n, n, bias=False).to(self.device)
            self.V_vel = nn.Linear(n, n, bias=False).to(self.device)

        # key, query, value matrices for adjacency space
        if self.adjacency:
            self.K_edge_attr = nn.Linear(
                N * (N - 1) * 2, N * (N - 1) * 2, bias=False
            ).to(self.device)
            self.Q_edge_attr = nn.Linear(
                N * (N - 1) * 2, N * (N - 1) * 2, bias=False
            ).to(self.device)
            self.V_edge_attr = nn.Linear(
                N * (N - 1) * 2, N * (N - 1) * 2, bias=False
            ).to(self.device)

        # example scalar hyperparameter
        self.B_const = 0.5
        # whether to use causal attention mask
        self.causal_attention = causal_attention
        # whether to use positional encoding
        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.positional_encoder = PositionalEncoder(
                B, N, L, d, n, kappa=kappa, dropout=dropout, device=device
            )

    def compute_vectorized_transform(self, xi, B_const):
        B, N, L, n = xi.size()

        # Difference matrix D_i = (B, N, L, 1, n) - (B, N, 1, L, n)
        D = xi.unsqueeze(3) - xi.unsqueeze(2)  # Shape: (B, N, L, L, n)

        # Compute squared differences
        squared_diffs = torch.sum(D**2, dim=-1)  # Shape: (B, N, L, L)

        if self.causal_attention:
            # Apply causal mask
            causal_mask = torch.triu(
                torch.ones(L, L), diagonal=1
            ).bool()  # Shape: (L, L)
            causal_mask = (
                causal_mask.unsqueeze(0).unsqueeze(0).to(xi.device)
            )  # Shape: (1, 1, L, L)

            # Apply mask to squared differences
            squared_diffs.masked_fill_(causal_mask, -float("inf"))

            # Compute beta_i using the softmax of the squared norm of the differences
            beta = F.softmax(squared_diffs, dim=-1)  # Shape: (B, N, L, L)
        else:
            # Compute beta_i using the softmax of the squared norm of the differences
            beta = F.softmax(squared_diffs, dim=-1)  # Shape: (B, N, L, L)

        # Define identity and ones matrices
        I = (
            torch.eye(L).unsqueeze(0).unsqueeze(0).to(self.device)
        )  # Shape: (1, 1, L, L)
        ones = torch.ones_like(I).to(self.device)  # Shape: (1, 1, L, L)

        # (1_{LxL} - I_{LxL}) element-wise multiplied by beta
        mask = ones - I
        masked_beta = mask * beta  # Shape: (B, N, L, L)

        # First term: xi
        xi_term = xi  # Shape: (B, N, L, n)

        # Check for term 2
        # calc1 = masked_beta[0, 0, :, :] @ xi_term[0, 0, :, :]
        # calc2 = torch.matmul(masked_beta, xi)[0, 0, :, :]
        # print(torch.equal(calc1, calc2))

        # Second term: B * (masked_beta @ xi)
        second_term = B_const * torch.matmul(masked_beta, xi)  # Shape: (B, N, L, n)

        # Third term: B * ((masked_beta @ ones) * xi)
        sum_masked_beta = torch.sum(
            masked_beta, dim=-1, keepdim=True
        )  # Shape: (B, N, L, 1)
        third_term = B_const * sum_masked_beta * xi  # Shape: (B, N, L, n)

        # Check for term 3
        # calc1 = xi[0, 0, :, :] * sum_masked_beta[0, 0, :, :]
        # calc2 = (sum_masked_beta * xi)[0, 0, :, :]
        # print(torch.equal(calc1, calc2))

        # Compute v
        v = xi_term + second_term - third_term  # Shape: (B, N, L, n)

        return v

    # theta corresponds to h (features), xi to x (positions), and omega to v (velocities)
    # theta has dim (B, N, L, d)
    # xi has dim (B, N, L, n)
    # omega has dim (B, N, L, n)
    # edge_attr has dim (B, L, N * (N-1) * 2)
    # B = batch size
    # N = num. atoms
    # L = seq length
    # n = dim of space, vel
    # d = feature space dim
    def forward(self, input):
        (theta, xi, omega, edge_attr) = input
        # theta = Variable(theta.data, requires_grad=True)
        # xi = Variable(xi.data, requires_grad=True)
        # omega = Variable(omega.data, requires_grad=True)

        # Positional encoder
        if self.positional_encoding:
            (theta, xi, omega, edge_attr) = self.positional_encoder(
                (theta, xi, omega, edge_attr)
            )

        # 1. Feature space
        key_features = self.K_features(theta)
        query_features = self.Q_features(theta)
        value_features = self.V_features(theta)

        h_out = F.scaled_dot_product_attention(
            query_features,
            key_features,
            value_features,
            is_causal=self.causal_attention,
        )

        # 2. Position space, equivariant to E(n)
        if self.equivariant:
            x_out = self.compute_vectorized_transform(xi, self.B_const)
        else:
            key_loc = self.K_loc(xi)
            query_loc = self.Q_loc(xi)
            value_loc = self.V_loc(xi)

            x_out = F.scaled_dot_product_attention(
                query_loc, key_loc, value_loc, is_causal=self.causal_attention
            )

        # 3. Velocity space, equivariant to SO(n)
        if self.equivariant:
            v_out = F.scaled_dot_product_attention(
                omega, omega, omega, is_causal=self.causal_attention
            )
        else:
            key_vel = self.K_vel(omega)
            query_vel = self.Q_vel(omega)
            value_vel = self.V_vel(omega)

            v_out = F.scaled_dot_product_attention(
                query_vel, key_vel, value_vel, is_causal=self.causal_attention
            )

        if self.adjacency:
            # 4. Edge attribute edge_attr is (B, L, N * (N - 1))
            key_edge_attr = self.K_edge_attr(edge_attr)
            query_edge_attr = self.Q_edge_attr(edge_attr)
            value_edge_attr = self.K_edge_attr(edge_attr)

            edge_attr_out = F.scaled_dot_product_attention(
                query_edge_attr,
                key_edge_attr,
                value_edge_attr,
                is_causal=self.causal_attention,
            )
            return (h_out, x_out, v_out, edge_attr_out)
        else:
            return (h_out, x_out, v_out, edge_attr)


class SpacetimeAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        spatial_attention=True,
        temporal_attention=True,
        positional_encoding=True,
        equivariant=True,
        causal_attention=True,
        dropout=0.1,
        kappa=10000,
        device="cpu",
        n_egnn_layers=4,
        recurrent=False,
        attention=True,
        norm_diff=False,
        tanh=False,
        adjacency=True,
        use_velocity=True,
        spatial_model_type="egnn",
    ):
        super(SpacetimeAttention, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.adjacency = adjacency
        self.positional_encoding = positional_encoding
        self.equivariant = equivariant
        self.causal_attention = causal_attention
        self.dropout = dropout
        self.kappa = kappa
        self.device = device
        self.use_velocity = use_velocity
        self.spatial_model_type = spatial_model_type

        # print(
        #     f"{batch_size}, {seq_len}, {num_atoms}, {in_node_nf}, {in_edge_nf}, {in_pos_nf}, {hidden_nf}, {proj_nf}"
        # )

        # spatial embedding model
        if spatial_attention:
            if self.spatial_model_type == "egnn":
                self.spatial_model = EGNN(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device="cpu",
                    n_layers=n_egnn_layers,
                )
            elif self.spatial_model_type == "egnnvel":
                self.spatial_model = EGNN_vel(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device=self.device,
                    n_layers=n_egnn_layers,
                    attention=attention,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                )
            elif self.spatial_model_type == "egnnschnet":
                self.spatial_model = EGNNSchnet(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device=self.device,
                    n_layers=n_egnn_layers,
                )
            elif self.spatial_model_type == "egnnvelschnet":
                self.spatial_model = EGNNvelSchnet(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device=self.device,
                    n_layers=n_egnn_layers,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                )

            # in_node_nf,
            # in_edge_nf,
            # hidden_nf,
            # proj_nf,
            # device="cpu",
            # act_fn=nn.SiLU(),
            # n_layers=4,
            # coords_weight=1.0,
            # hidden_channels=32,
            # num_filters=32,
            # num_interactions=6,
            # cutoff=10.0,

        if self.temporal_attention:
            # temporal attention model
            self.temporal_model = ETAL_vel(
                self.B,
                self.N,
                self.L,
                self.d,
                self.n,
                dropout=self.dropout,
                kappa=self.kappa,
                adjacency=self.adjacency,
                positional_encoding=self.positional_encoding,
                equivariant=self.equivariant,
                causal_attention=self.causal_attention,
                device=self.device,
            )

            # if self.use_velocity:
            #     self.temporal_model = ETAL_vel(
            #         self.B,
            #         self.N,
            #         self.L,
            #         self.d,
            #         self.n,
            #         dropout=self.dropout,
            #         kappa=self.kappa,
            #         adjacency=self.adjacency,
            #         positional_encoding=self.positional_encoding,
            #         equivariant=self.equivariant,
            #         causal_attention=self.causal_attention,
            #         device=self.device,
            #     )
            # else:
            #     self.temporal_model = ETAL(
            #         self.B,
            #         self.N,
            #         self.L,
            #         self.d,
            #         self.n,
            #         dropout=self.dropout,
            #         kappa=self.kappa,
            #         adjacency=self.adjacency,
            #         positional_encoding=self.positional_encoding,
            #         equivariant=self.equivariant,
            #         causal_attention=self.causal_attention,
            #         device=self.device,
            #     )

    def forward(self, nodes, loc, edges, vel, edge_attr, batch=None):
        if self.spatial_attention:
            # Apply EGCL to each graph 'token' for t=1,...,L
            # Here, we effectively share the same EGCL--this is like an embedding in the usual sense of the Transformer
            h_spatial, loc_spatial, vel_spatial = self.spatial_model(
                nodes, loc, edges, vel, edge_attr
            )

            h_spatial = h_spatial.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.L, self.n)
            if self.use_velocity:
                vel_spatial = vel_spatial.reshape(self.B, self.N, self.L, self.n)
        else:
            h_spatial = nodes.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc.reshape(self.B, self.N, self.L, self.n)
            if self.use_velocity:
                vel_spatial = vel.reshape(self.B, self.N, self.L, self.n)

        if self.temporal_attention:
            if not (edge_attr is None):
                # edge_attr is (B * N * (N - 1) * L, 2), want (B, L, N * (N - 1) * 2) for temporal model
                # (B * N * (N - 1) * L, 2) -> (B, N * (N - 1), L, 2)
                edge_attr_spatial = edge_attr.reshape(
                    self.B, self.N * (self.N - 1), self.L, 2
                )
                # (B, N * (N - 1), L, 2) -> (B, L, N * (N - 1), 2)
                edge_attr_spatial = edge_attr_spatial.transpose(1, 2)
                # (B, L, N * (N - 1), 2) -> (B, L, N * (N - 1) * 2)
                edge_attr_spatial = edge_attr_spatial.reshape(
                    self.B, self.L, self.N * (self.N - 1) * 2
                )

            # APPLY TEMPORAL ATTENTION
            # h_spatial/h_temporal is (B, N, L, d)
            # loc_spatial/loc_temporal is (B, N, L, n)
            # vel_spatial/vel_temporal is (B, N, L, n)
            # edge_attr_spatial/edge_attr_temporal is (B, L, N * (N - 1) * 2)
            if not self.spatial_attention:
                h_spatial = torch.tensor(h_spatial.data, requires_grad=True)
                loc_spatial = torch.tensor(loc_spatial.data, requires_grad=True)
                if self.use_velocity:
                    vel_spatial = torch.tensor(vel_spatial.data, requires_grad=True)
                if not (edge_attr is None):
                    edge_attr_spatial = torch.tensor(
                        edge_attr_spatial.data, requires_grad=True
                    )

            h_temporal, loc_temporal, vel_temporal, edge_attr_temporal = (
                self.temporal_model(
                    (h_spatial, loc_spatial, vel_spatial, edge_attr_spatial)
                )
            )
            # if self.use_velocity:
            #     h_temporal, loc_temporal, vel_temporal, edge_attr_temporal = (
            #         self.temporal_model(
            #             (h_spatial, loc_spatial, vel_spatial, edge_attr_spatial)
            #         )
            #     )
            # else:
            #     h_temporal, loc_temporal, edge_attr_temporal = self.temporal_model(
            #         (h_spatial, loc_spatial, edge_attr_spatial)
            #     )

            # (B, N, L, d) -> (B * N * L, d)
            h_temporal = h_temporal.reshape(self.B * self.N * self.L, self.d)
            # (B, N, L, n) -> (B * N * L, n)
            loc_temporal = loc_temporal.reshape(self.B * self.N * self.L, self.n)
            # (B, N, L, n) -> (B * N * L, n)
            if self.use_velocity:
                vel_temporal = vel_temporal.reshape(self.B * self.N * self.L, self.n)
            # Want: (B, L, N * (N - 1) * 2) -> (B * N * (N - 1) * L, 2)
            # (B, L, N * (N - 1) * 2) -> (B, L, N * (N - 1), 2)
            if not (edge_attr is None):
                edge_attr_temporal = edge_attr_temporal.reshape(
                    self.B, self.L, self.N * (self.N - 1), 2
                )
                # (B, L, N * (N - 1), 2) -> (B, N * (N - 1), L, 2)
                edge_attr_temporal = edge_attr_temporal.transpose(1, 2)
                # (B, N * (N - 1), L, 2) -> (B * N * (N - 1) * L, 2)
                edge_attr_temporal = edge_attr_temporal.reshape(
                    self.B * self.N * (self.N - 1) * self.L, 2
                )

            # Take a mean across the time dimension for h, x, v
            # to get dim (B, N, d) and (B, N, n)
            # h_temporal = h_temporal.mean(dim=-2)
            # loc_temporal = loc_temporal.mean(dim=-2)
            # vel_temporal = vel_temporal.mean(dim=-2)

            # Or just extract the last time component
            # h_temporal_pred = h_temporal_pred[:, :, -1, :]
            # loc_temporal_pred = loc_temporal_pred[:, :, -1, :]
            # vel_temporal_pred = vel_temporal_pred[:, :, -1, :]
            if self.use_velocity:
                return h_temporal, loc_temporal, vel_temporal, edge_attr_temporal
            else:
                return h_temporal, loc_temporal, edge_attr_temporal
        else:
            # (B, N, L, d) -> (B * N * L, d)
            h_spatial = h_spatial.reshape(self.B * self.N * self.L, self.d)
            # (B, N, L, n) -> (B * N * L, n)
            loc_spatial = loc_spatial.reshape(self.B * self.N * self.L, self.n)
            if self.use_velocity:
                # (B, N, L, n) -> (B * N * L, n)
                vel_spatial = vel_spatial.reshape(self.B * self.N * self.L, self.n)
                return h_spatial, loc_spatial, vel_spatial, edge_attr
            else:
                return h_spatial, loc_spatial, edge_attr


class SpacetimeTransformer(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        spatial_attention=True,
        temporal_attention=True,
        positional_encoding=True,
        causal_attention=True,
        dropout=0.1,
        kappa=10000,
        equivariant=True,
        device="cpu",
        n_egnn_layers=4,
        n_stacking_layers=5,
        recurrent=False,
        attention=True,
        norm_diff=False,
        tanh=False,
        adjacency=True,
        use_velocity=True,
        spatial_model_type="egnn",
    ):
        super(SpacetimeTransformer, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.adjacency = adjacency
        self.n_stacking_layers = n_stacking_layers
        self.equivariant = equivariant
        self.use_velocity = use_velocity
        self.spatial_model_type = spatial_model_type

        for i in range(0, self.n_stacking_layers):
            self.add_module(
                "spacetime_attention_%d" % i,
                SpacetimeAttention(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_atoms=num_atoms,
                    in_node_nf=in_node_nf,
                    in_edge_nf=in_edge_nf,
                    in_pos_nf=in_pos_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    spatial_attention=spatial_attention,
                    temporal_attention=self.temporal_attention,
                    positional_encoding=positional_encoding,
                    equivariant=equivariant,
                    causal_attention=causal_attention,
                    dropout=dropout,
                    kappa=kappa,
                    device=device,
                    n_egnn_layers=n_egnn_layers,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    adjacency=adjacency,
                    use_velocity=use_velocity,
                ),
            )

        # layer normalization for i = 1, ..., n_stacking_layers - 1
        for i in range(0, self.n_stacking_layers - 1):
            self.add_module(
                "h_layer_norm_%d" % i, nn.LayerNorm([self.d], elementwise_affine=False)
            )
            self.add_module(
                "edge_attr_layer_norm_%d" % i,
                nn.LayerNorm([2], elementwise_affine=False),
            )
            if not self.equivariant:
                self.add_module(
                    "loc_layer_norm_%d" % i,
                    nn.LayerNorm([self.n], elementwise_affine=False),
                )
                if self.use_velocity:
                    self.add_module(
                        "vel_layer_norm_%d" % i,
                        nn.LayerNorm([self.n], elementwise_affine=False),
                    )

        # MLPs
        self.h_mlp = nn.Linear(self.d, self.d)
        self.edge_attr_mlp = nn.Linear(2, 2)
        if not self.equivariant:
            self.loc_mlp = nn.Linear(self.n, self.n)
            if self.use_velocity:
                self.vel_mlp = nn.Linear(self.n, self.n)

    def forward(
        self,
        h_spatiotemporal,
        loc_spatiotemporal,
        edges,
        vel_spatiotemporal=None,
        edge_attr_spatiotemporal=None,
        batch=None,
    ):
        # print(
        #     f"h_spatiotemporal {h_spatiotemporal.shape}, loc_spatiotemporal {loc_spatiotemporal.shape}, vel_spatiotemporal {vel_spatiotemporal.shape}, edge_attr_spatiotemporal {edge_attr_spatiotemporal.shape}"
        # )

        for i in range(0, self.n_stacking_layers):
            # Apply spacetime attention
            if self.spatial_model_type == "egnn":
                (
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edge_attr_spatiotemporal,
                ) = self._modules["spacetime_attention_%d" % i](
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edges,
                    vel_spatiotemporal,
                    edge_attr=edge_attr_spatiotemporal,
                )
            elif self.spatial_model_type == "egnnvel":
                (
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    vel_spatiotemporal,
                    edge_attr_spatiotemporal,
                ) = self._modules["spacetime_attention_%d" % i](
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edges,
                    vel_spatiotemporal,
                    edge_attr=edge_attr_spatiotemporal,
                )
            elif self.spatial_model_type == "egnnschnet":
                (
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    vel_spatiotemporal,
                    edge_attr_spatiotemporal,
                ) = self._modules["spacetime_attention_%d" % i](
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edges,
                    vel_spatiotemporal,
                    edge_attr=edge_attr_spatiotemporal,
                    batch=batch,
                )
            elif self.spatial_model_type == "egnnvelschnet":
                (
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    vel_spatiotemporal,
                    edge_attr_spatiotemporal,
                ) = self._modules["spacetime_attention_%d" % i](
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edges,
                    vel_spatiotemporal,
                    edge_attr=edge_attr_spatiotemporal,
                    batch=batch,
                )

            # layer normalization is applied to all layers except the last
            if i < self.n_stacking_layers - 1:
                h_spatiotemporal = self._modules["h_layer_norm_%d" % i](
                    h_spatiotemporal
                )
                if not (edge_attr_spatiotemporal is None):
                    edge_attr_spatiotemporal = self._modules[
                        "edge_attr_layer_norm_%d" % i
                    ](edge_attr_spatiotemporal)
                if not self.equivariant:
                    # Apply layer normalization if not equivariant
                    loc_spatiotemporal = self._modules["loc_layer_norm_%d" % i](
                        loc_spatiotemporal
                    )
                    if self.use_velocity:
                        vel_spatiotemporal = self._modules["vel_layer_norm_%d" % i](
                            vel_spatiotemporal
                        )
                # else:
                #     # Just subtract the mean over the final dimension if equivariant
                #     loc_spatiotemporal = loc_spatiotemporal - loc_spatiotemporal.mean(
                #         -1
                #     ).unsqueeze(-1)
                #     if self.use_velocity:
                #         vel_spatiotemporal = (
                #             vel_spatiotemporal
                #             - vel_spatiotemporal.mean(-1).unsqueeze(-1)
                #         )

            # Apply MLP and residual to h and edge_attr
            h_spatiotemporal = self.h_mlp(h_spatiotemporal) + h_spatiotemporal
            if not (edge_attr_spatiotemporal is None):
                edge_attr_spatiotemporal = (
                    self.edge_attr_mlp(edge_attr_spatiotemporal)
                    + edge_attr_spatiotemporal
                )

            # Only apply MLP and residual to other components if equivariance is to be ignored
            if not self.equivariant:
                loc_spatiotemporal = (
                    self.loc_mlp(loc_spatiotemporal) + loc_spatiotemporal
                )
                if self.use_velocity:
                    vel_spatiotemporal = (
                        self.vel_mlp(vel_spatiotemporal) + vel_spatiotemporal
                    )

        # h_spatiotemporal is (B * N * L, d)
        # loc_spatiotemporal is (B * N * L, n)
        # vel_spatiotemporal is (B * N * L, n)
        # edge_attr_spatiotemporal is (B * N * (N - 1) * L, 2)

        # As this is the final layer, we would like to compute means
        # (B * N * L, d) -> (B, N, L, d)
        h_spatiotemporal = h_spatiotemporal.reshape(self.B, self.N, self.L, self.d)
        # (B, N, L, d) -> (B, N, d)
        h_spatiotemporal = h_spatiotemporal.mean(-2)

        # (B * N * L, n) -> (B, N, L, n)
        loc_spatiotemporal = loc_spatiotemporal.reshape(self.B, self.N, self.L, self.n)
        # (B, N, L, n) -> (B, N, n)
        loc_spatiotemporal = loc_spatiotemporal.mean(-2)

        if self.use_velocity:
            # (B * N * L, n) -> (B, N, L, n)
            vel_spatiotemporal = vel_spatiotemporal.reshape(
                self.B, self.N, self.L, self.n
            )
            # (B, N, L, n) -> (B, N, n)
            vel_spatiotemporal = vel_spatiotemporal.mean(-2)
        return (h_spatiotemporal, loc_spatiotemporal, vel_spatiotemporal)
        # else:
        #     return (h_spatiotemporal, loc_spatiotemporal)


class SpacetimeLSTM(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        spatial_attention=True,
        dropout=0.1,
        device="cpu",
        n_egnn_layers=4,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super(SpacetimeLSTM, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.spatial_attention = spatial_attention
        self.dropout = dropout

        # print(
        #     f"{batch_size}, {seq_len}, {num_atoms}, {in_node_nf}, {in_edge_nf}, {in_pos_nf}, {hidden_nf}, {proj_nf}"
        # )

        # spatial embedding model
        self.spatial_model = EGNN_vel(
            in_node_nf=self.d,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            proj_nf=proj_nf,
            device=device,
            n_layers=n_egnn_layers,
            # attention=attention,
            recurrent=recurrent,
            norm_diff=norm_diff,
            tanh=tanh,
        )
        # temporal attention model
        input_nf = self.N * (self.d + 2 * self.n)
        # self.temporal_model = nn.LSTM(
        #     input_nf, hidden_nf * 2, proj_size=input_nf, batch_first=True
        # )
        self.temporal_model = nn.LSTM(input_nf, input_nf, batch_first=True)

    def forward(self, nodes, loc, edges, vel, edge_attr):
        if self.spatial_attention:
            # Apply EGCL to each graph 'token' for t=1,...,L
            # Here, we effectively share the same EGCL--this is like an embedding in the usual sense of the Transformer
            h_spatial, loc_spatial, vel_spatial = self.spatial_model(
                nodes, loc, edges, vel, edge_attr
            )
            # print(
            #     f"nodes {nodes.shape} loc {loc.shape} vel {vel.shape} edge attr {edge_attr.shape}"
            # )

            h_spatial = h_spatial.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.L, self.n)
            vel_spatial = vel_spatial.reshape(self.B, self.N, self.L, self.n)
        else:
            h_spatial = nodes.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc.reshape(self.B, self.N, self.L, self.n)
            vel_spatial = vel.reshape(self.B, self.N, self.L, self.n)

        # (B, N, L, d) -> (B, L, N, d)
        h_spatial = h_spatial.transpose(1, 2)
        # (B, L, N, d) -> (B, L, N * d)
        h_spatial = h_spatial.reshape(self.B, self.L, self.N * self.d)

        # (B, N, L, n) -> (B, L, N, n)
        loc_spatial = loc_spatial.transpose(1, 2)
        # (B, L, N, n) -> (B, L, N * n)
        loc_spatial = loc_spatial.reshape(self.B, self.L, self.N * self.n)

        # (B, N, L, n) -> (B, L, N, n)
        vel_spatial = vel_spatial.transpose(1, 2)
        # (B, L, N, n) -> (B, L, N * n)
        vel_spatial = vel_spatial.reshape(self.B, self.L, self.N * self.n)

        # # edge_attr is (B * N * (N - 1) * L, 2), want (B, L, N * (N - 1) * 2) for temporal model
        # # (B * N * (N - 1) * L, 2) -> (B, N * (N - 1), L, 2)
        # edge_attr_spatial = edge_attr.reshape(self.B, self.N * (self.N - 1), self.L, 2)
        # # (B, N * (N - 1), L, 2) -> (B, L, N * (N - 1), 2)
        # edge_attr_spatial = edge_attr_spatial.transpose(1, 2)
        # # (B, L, N * (N - 1), 2) -> (B, L, N * (N - 1) * 2)
        # edge_attr_spatial = edge_attr_spatial.reshape(
        #     self.B, self.L, self.N * (self.N - 1) * 2
        # )

        if not self.spatial_attention:
            h_spatial = torch.tensor(h_spatial.data, requires_grad=True)
            loc_spatial = torch.tensor(loc_spatial.data, requires_grad=True)
            vel_spatial = torch.tensor(vel_spatial.data, requires_grad=True)
            edge_attr_spatial = torch.tensor(edge_attr_spatial.data, requires_grad=True)

        # APPLY TEMPORAL LSTM
        # h_spatial is (B, L, N * d)
        # loc_spatial is (B, L, N * n)
        # vel_spatial is (B, L, N * n)
        # edge_attr_spatial is (B, L, N * (N - 1) * 2)

        # concatenate input across last dimension to get (B, L, N(d + 2n)), to feed to LSTM
        input_spatial = torch.cat((h_spatial, loc_spatial, vel_spatial), -1)
        # (B, L, N(d + 2n))
        output_temporal, _ = self.temporal_model(input_spatial)
        # unconcatenate/roll out
        h_temporal = output_temporal[:, :, : self.N * self.d]
        loc_temporal = output_temporal[
            :, :, self.N * self.d : self.N * self.d + self.N * self.n
        ]
        vel_temporal = output_temporal[:, :, self.N * self.d + self.N * self.n :]
        # edge_attr_temporal = output_temporal[
        #     :, :, self.N * self.d + 2 * self.N * self.n :
        # ]
        edge_attr_temporal = edge_attr

        # print("VERIFICATION")
        # print(f"h {h_temporal.shape}")
        # print(f"loc {loc_temporal.shape}")
        # print(f"vel {vel_temporal.shape}")
        # print(f"edge attr {edge_attr_temporal.shape}")
        # breakpoint()

        # (B, L, Nd) -> (B, L, N, d)
        h_temporal = h_temporal.reshape(self.B, self.L, self.N, self.d)
        # (B, L, N, d) -> (B, N, L, d)
        h_temporal = h_temporal.transpose(1, 2)
        # (B, N, L, d) -> (B * N * L, d)
        h_temporal = h_temporal.reshape(self.B * self.N * self.L, self.d)

        # (B, L, Nn) -> (B, L, N, n)
        loc_temporal = loc_temporal.reshape(self.B, self.L, self.N, self.n)
        # (B, L, N, n) -> (B, N, L, n)
        loc_temporal = loc_temporal.transpose(1, 2)
        # (B, N, L, n) -> (B * N * L, n)
        loc_temporal = loc_temporal.reshape(self.B * self.N * self.L, self.n)

        # (B, L, Nn) -> (B, L, N, n)
        vel_temporal = vel_temporal.reshape(self.B, self.L, self.N, self.n)
        # (B, L, N, n) -> (B, N, L, n)
        vel_temporal = vel_temporal.transpose(1, 2)
        # (B, N, L, n) -> (B * N * L, n)
        vel_temporal = vel_temporal.reshape(self.B * self.N * self.L, self.n)

        # # Want: (B, L, N * (N - 1) * 2) -> (B * N * (N - 1) * L, 2)
        # # (B, L, N * (N - 1) * 2) -> (B, L, N * (N - 1), 2)
        # edge_attr_temporal = edge_attr_temporal.reshape(
        #     self.B, self.L, self.N * (self.N - 1), 2
        # )
        # # (B, L, N * (N - 1), 2) -> (B, N * (N - 1), L, 2)
        # edge_attr_temporal = edge_attr_temporal.transpose(1, 2)
        # # (B, N * (N - 1), L, 2) -> (B * N * (N - 1) * L, 2)
        # edge_attr_temporal = edge_attr_temporal.reshape(
        #     self.B * self.N * (self.N - 1) * self.L, 2
        # )

        # Take a mean across the time dimension for h, x, v
        # to get dim (B, N, d) and (B, N, n)
        # h_temporal = h_temporal.mean(dim=-2)
        # loc_temporal = loc_temporal.mean(dim=-2)
        # vel_temporal = vel_temporal.mean(dim=-2)

        # Or just extract the last time component
        # h_temporal_pred = h_temporal_pred[:, :, -1, :]
        # loc_temporal_pred = loc_temporal_pred[:, :, -1, :]
        # vel_temporal_pred = vel_temporal_pred[:, :, -1, :]

        return h_temporal, loc_temporal, vel_temporal, edge_attr_temporal


class SpacetimeLSTMStacked(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        spatial_attention=True,
        dropout=0.1,
        device="cpu",
        n_egnn_layers=4,
        n_stacking_layers=5,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super(SpacetimeLSTMStacked, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.spatial_attention = spatial_attention
        self.n_stacking_layers = n_stacking_layers

        for i in range(0, self.n_stacking_layers):
            self.add_module(
                "spacetime_lstm_%d" % i,
                SpacetimeLSTM(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    num_atoms=num_atoms,
                    in_node_nf=in_node_nf,
                    in_edge_nf=in_edge_nf,
                    in_pos_nf=in_pos_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    spatial_attention=True,
                    dropout=dropout,
                    device=device,
                    n_egnn_layers=n_egnn_layers,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )

        # layer normalization for i = 1, ..., n_stacking_layers - 1
        for i in range(0, self.n_stacking_layers - 1):
            self.add_module(
                "h_layer_norm_%d" % i, nn.LayerNorm([self.d], elementwise_affine=False)
            )
            self.add_module(
                "edge_attr_layer_norm_%d" % i,
                nn.LayerNorm([2], elementwise_affine=False),
            )
            self.add_module(
                "loc_layer_norm_%d" % i,
                nn.LayerNorm([self.n], elementwise_affine=False),
            )
            self.add_module(
                "vel_layer_norm_%d" % i,
                nn.LayerNorm([self.n], elementwise_affine=False),
            )

        # MLPs
        self.h_mlp = nn.Linear(self.d, self.d)
        self.edge_attr_mlp = nn.Linear(2, 2)
        self.loc_mlp = nn.Linear(self.n, self.n)
        self.vel_mlp = nn.Linear(self.n, self.n)

    def forward(
        self,
        h_spatiotemporal,
        loc_spatiotemporal,
        edges,
        vel_spatiotemporal,
        edge_attr_spatiotemporal,
    ):
        # print(
        #     f"h_spatiotemporal {h_spatiotemporal.shape}, loc_spatiotemporal {loc_spatiotemporal.shape}, vel_spatiotemporal {vel_spatiotemporal.shape}, edge_attr_spatiotemporal {edge_attr_spatiotemporal.shape}"
        # )

        for i in range(0, self.n_stacking_layers):
            # Apply spacetime attention
            (
                h_spatiotemporal,
                loc_spatiotemporal,
                vel_spatiotemporal,
                edge_attr_spatiotemporal,
            ) = self._modules["spacetime_lstm_%d" % i](
                h_spatiotemporal,
                loc_spatiotemporal,
                edges,
                vel_spatiotemporal,
                edge_attr=edge_attr_spatiotemporal,
            )

            # layer normalization is applied to all layers except the last
            if i < self.n_stacking_layers - 1:
                h_spatiotemporal = self._modules["h_layer_norm_%d" % i](
                    h_spatiotemporal
                )
                loc_spatiotemporal = self._modules["loc_layer_norm_%d" % i](
                    loc_spatiotemporal
                )
                vel_spatiotemporal = self._modules["vel_layer_norm_%d" % i](
                    vel_spatiotemporal
                )
                edge_attr_spatiotemporal = self._modules["edge_attr_layer_norm_%d" % i](
                    edge_attr_spatiotemporal
                )

            # Apply MLP and residual to h, loc, vel, edge_attr
            h_spatiotemporal = self.h_mlp(h_spatiotemporal) + h_spatiotemporal
            loc_spatiotemporal = self.loc_mlp(loc_spatiotemporal) + loc_spatiotemporal
            vel_spatiotemporal = self.vel_mlp(vel_spatiotemporal) + vel_spatiotemporal
            edge_attr_spatiotemporal = (
                self.edge_attr_mlp(edge_attr_spatiotemporal) + edge_attr_spatiotemporal
            )

        # h_spatiotemporal is (B * N * L, d)
        # loc_spatiotemporal is (B * N * L, n)
        # vel_spatiotemporal is (B * N * L, n)
        # edge_attr_spatiotemporal is (B * N * (N - 1) * L, 2)

        # As this is the final layer, we would like to compute means
        # (B * N * L, d) -> (B, N, L, d)
        h_spatiotemporal = h_spatiotemporal.reshape(self.B, self.N, self.L, self.d)
        # (B, N, L, d) -> (B, N, d)
        h_spatiotemporal = h_spatiotemporal.mean(-2)

        # (B * N * L, n) -> (B, N, L, n)
        loc_spatiotemporal = loc_spatiotemporal.reshape(self.B, self.N, self.L, self.n)
        # (B, N, L, n) -> (B, N, n)
        loc_spatiotemporal = loc_spatiotemporal.mean(-2)

        # (B * N * L, n) -> (B, N, L, n)
        vel_spatiotemporal = vel_spatiotemporal.reshape(self.B, self.N, self.L, self.n)
        # (B, N, L, n) -> (B, N, n)
        vel_spatiotemporal = vel_spatiotemporal.mean(-2)

        return (h_spatiotemporal, loc_spatiotemporal, vel_spatiotemporal)


class RF_vel(nn.Module):
    def __init__(
        self, hidden_nf, edge_attr_nf=0, device="cpu", act_fn=nn.SiLU(), n_layers=4
    ):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        # self.reg = reg
        ### Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn),
            )
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x


class EGNNvelBaseline(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        time_series,
        device,
        n_egnn_layers,
        attention,
        recurrent,
        norm_diff,
        tanh,
    ):
        super(EGNNvelBaseline, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.time_series = time_series

        self.spatial_model = EGNN_vel(
            in_node_nf=self.d,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            proj_nf=proj_nf,
            device=device,
            n_layers=n_egnn_layers,
            attention=attention,
            recurrent=recurrent,
            norm_diff=norm_diff,
            tanh=tanh,
        )

    def forward(self, nodes, loc, edges, vel, edge_attr):
        # Apply EGCL to each graph 'token' for t=1,...,L
        # Here, we effectively share the same EGCL--this is like an embedding in the usual sense of the Transformer
        h_spatial, loc_spatial, vel_spatial = self.spatial_model(
            nodes, loc, edges, vel, edge_attr
        )
        # print(
        #     f"nodes {nodes.shape} loc {loc.shape} vel {vel.shape} edge attr {edge_attr.shape}"
        # )

        if self.time_series:
            h_spatial = h_spatial.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.L, self.n)
            vel_spatial = vel_spatial.reshape(self.B, self.N, self.L, self.n)

            # h_spatial is (B, N, L, d)
            # loc_spatial is (B, N, L, n)
            # vel_spatial is (B, N, L, n)

            # As this is the final layer, we would like to compute means
            # (B, N, L, d) -> (B, N, d)
            h_spatial = h_spatial.mean(-2)
            # (B, N, L, n) -> (B, N, n)
            loc_spatial = loc_spatial.mean(-2)
            # (B, N, L, n) -> (B, N, n)
            vel_spatial = vel_spatial.mean(-2)
        else:
            h_spatial = h_spatial.reshape(self.B, self.N, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.n)
            vel_spatial = vel_spatial.reshape(self.B, self.N, self.n)

        return (h_spatial, loc_spatial, vel_spatial)


class EGNNvelSchnetBaseline(nn.Module):
    def __init__(
        self,
        batch_size,
        seq_len,
        num_atoms,
        in_node_nf,
        in_edge_nf,
        in_pos_nf,
        hidden_nf,
        proj_nf,
        time_series,
        device,
        n_egnn_layers,
        recurrent,
        attention,
        norm_diff,
        tanh,
    ):
        super(EGNNvelSchnetBaseline, self).__init__()
        # in_node_nf is d=1 (or dim of h)
        # in_edge_nf is 2d=2
        # in_pos_nf is n=3
        # hidden_nf is 64
        # proj_nf is 1 (what we project h to in EGNN)
        # batch_size (B)
        # num_atoms (N)
        # seq_len (L)
        # h_dim (d)
        # x_dim, v_dim (n)
        self.d = in_node_nf
        self.B = batch_size
        self.n = in_pos_nf
        self.L = seq_len
        self.N = num_atoms
        self.time_series = time_series

        self.spatial_model = EGNNvelSchnet(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_nf=hidden_nf,
            proj_nf=proj_nf,
            device=device,
            n_layers=n_egnn_layers,
            recurrent=recurrent,
            attention=attention,
            norm_diff=norm_diff,
            tanh=tanh,
        )

    def forward(self, nodes, loc, edges, vel, edge_attr, batch):
        # Apply EGCL to each graph 'token' for t=1,...,L
        # Here, we effectively share the same EGCL--this is like an embedding in the usual sense of the Transformer
        h_spatial, loc_spatial, vel_spatial = self.spatial_model(
            nodes, loc, edges, vel, edge_attr, batch
        )
        # print(
        #     f"nodes {nodes.shape} loc {loc.shape} vel {vel.shape} edge attr {edge_attr.shape}"
        # )

        if self.time_series:
            # h_spatial = h_spatial.reshape(self.B, self.N, self.L, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.L, self.n)
            vel_spatial = vel_spatial.reshape(self.B, self.N, self.L, self.n)

            # h_spatial is (B, N, L, d)
            # loc_spatial is (B, N, L, n)
            # vel_spatial is (B, N, L, n)

            # As this is the final layer, we would like to compute means
            # (B, N, L, d) -> (B, N, d)
            # h_spatial = h_spatial.mean(-2)
            # (B, N, L, n) -> (B, N, n)
            loc_spatial = loc_spatial.mean(-2)
            # (B, N, L, n) -> (B, N, n)
            vel_spatial = vel_spatial.mean(-2)
        else:
            # h_spatial = h_spatial.reshape(self.B, self.N, self.d)
            loc_spatial = loc_spatial.reshape(self.B, self.N, self.n)
            vel_spatial = vel_spatial.reshape(self.B, self.N, self.n)

        return (None, loc_spatial, vel_spatial)


from se3_transformer_pytorch import SE3Transformer


class SE3TransformerBaseline(nn.Module):
    def __init__(self, dim, heads, depth, dim_head, num_degrees, valid_radius, proj_nf):
        super(SE3TransformerBaseline, self).__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.num_degrees = num_degrees
        self.valid_radius = valid_radius

        self.transformer = SE3Transformer(
            dim=dim,
            heads=heads,
            depth=depth,
            dim_head=dim_head,
            num_degrees=num_degrees,
            valid_radius=valid_radius,
        )

        self.projection = nn.Linear(dim, proj_nf)

    def forward(self, h, x):
        x = self.transformer(h, x)
        x = self.projection(x)

        return x
