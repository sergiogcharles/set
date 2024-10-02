import torch
from torch import nn
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel
from torch.nn import functional as F
import math


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

    def forward(self, h, x, edges, edge_attr, vel=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            # if vel is not None:
            # vel_attr = get_velocity_attr(x, vel, edges[0], edges[1])
            # edge_attr = torch.cat([edge_attr0, vel_attr], dim=1).detach()
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)

        h = self.projection(h)

        return h, x


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
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                ),
            )
        self.projection = nn.Linear(self.hidden_nf, self.proj_nf)

        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, vel, _ = self._modules["gcl_%d" % i](
                h, edges, x, vel, edge_attr=edge_attr
            )
        # project down
        h = self.projection(h)

        return h, x, vel


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
        norm_diff=False,
        tanh=False,
        adjacency=True,
        use_velocity=True,
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

        # print(
        #     f"{batch_size}, {seq_len}, {num_atoms}, {in_node_nf}, {in_edge_nf}, {in_pos_nf}, {hidden_nf}, {proj_nf}"
        # )

        # spatial embedding model
        if spatial_attention:
            if self.use_velocity:
                self.spatial_model = EGNN_vel(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device=self.device,
                    n_layers=n_egnn_layers,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                )
            else:
                self.spatial_model = EGNN(
                    in_node_nf=self.d,
                    in_edge_nf=in_edge_nf,
                    hidden_nf=hidden_nf,
                    proj_nf=proj_nf,
                    device="cpu",
                    n_layers=n_egnn_layers,
                )

        if self.temporal_attention:
            # temporal attention model
            if self.use_velocity:
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
            else:
                self.temporal_model = ETAL(
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

    def forward(self, nodes, loc, edges, vel, edge_attr):
        if self.spatial_attention:
            # Apply EGCL to each graph 'token' for t=1,...,L
            # Here, we effectively share the same EGCL--this is like an embedding in the usual sense of the Transformer
            if self.use_velocity:
                h_spatial, loc_spatial, vel_spatial = self.spatial_model(
                    nodes, loc, edges, vel, edge_attr
                )
            else:
                h_spatial, loc_spatial = self.spatial_model(
                    nodes, loc, edges, edge_attr
                )
            # print(
            #     f"nodes {nodes.shape} loc {loc.shape} vel {vel.shape} edge attr {edge_attr.shape}"
            # )

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

            if self.use_velocity:
                h_temporal, loc_temporal, vel_temporal, edge_attr_temporal = (
                    self.temporal_model(
                        (h_spatial, loc_spatial, vel_spatial, edge_attr_spatial)
                    )
                )
            else:
                h_temporal, loc_temporal, edge_attr_temporal = self.temporal_model(
                    (h_spatial, loc_spatial, edge_attr_spatial)
                )

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
        norm_diff=False,
        tanh=False,
        adjacency=True,
        use_velocity=True,
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
    ):
        # print(
        #     f"h_spatiotemporal {h_spatiotemporal.shape}, loc_spatiotemporal {loc_spatiotemporal.shape}, vel_spatiotemporal {vel_spatiotemporal.shape}, edge_attr_spatiotemporal {edge_attr_spatiotemporal.shape}"
        # )

        for i in range(0, self.n_stacking_layers):
            # Apply spacetime attention
            if self.use_velocity:
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
            else:
                (
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edge_attr_spatiotemporal,
                ) = self._modules["spacetime_attention_%d" % i](
                    h_spatiotemporal,
                    loc_spatiotemporal,
                    edges,
                    edge_attr=edge_attr_spatiotemporal,
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
                else:
                    # Just subtract the mean over the final dimension if equivariant
                    loc_spatiotemporal = loc_spatiotemporal - loc_spatiotemporal.mean(
                        -1
                    ).unsqueeze(-1)
                    if self.use_velocity:
                        vel_spatiotemporal = (
                            vel_spatiotemporal
                            - vel_spatiotemporal.mean(-1).unsqueeze(-1)
                        )

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
        else:
            return (h_spatiotemporal, loc_spatiotemporal)


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


class Baseline(nn.Module):
    def __init__(self, device="cpu"):
        super(Baseline, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc


class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device="cpu"):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)


class MLP_dynamics(nn.Module):
    def __init__(self, n_layers, input_nf, output_nf, hidden_nf, device="cpu"):
        super(MLP_dynamics, self).__init__()
        self.output_nf = output_nf
        layers = []

        # Input layer
        layers.append(nn.Linear(2 * input_nf, hidden_nf))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_nf, hidden_nf))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_nf, 2 * output_nf))

        # Combine all layers into a Sequential module
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, v):
        xv = torch.cat((x, v), -1)
        out = self.mlp(xv)
        x_out = out[:, :, :, : self.output_nf]
        v_out = out[:, :, :, self.output_nf :]
        return x_out, v_out


class Linear_dynamics(nn.Module):
    def __init__(self, device="cpu"):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1) * 1)
        self.delta_v = nn.Parameter(torch.ones(1) * 0.0)
        self.slope_v = nn.Parameter(torch.ones(1) * 1.0)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v * self.time, self.slope_v * v + self.delta_v


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
        recurrent,
        norm_diff,
        tanh,
    ):
        super(EGNNBaseline, self).__init__()
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
