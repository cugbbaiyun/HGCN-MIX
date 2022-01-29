import torch
from torch._C import INSERT_FOLD_PREPACK_OPS, Node
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class HGCN(nn.Module):
    def __init__(self, n_edges, in_feature, out_feature, n_agents):
        super(HGCN, self).__init__()
        print(n_edges)
        self.W_line = nn.Parameter(torch.ones(n_edges).cuda())
        self.W = None

    def forward(self, node_features, hyper_graph):
        self.W = torch.diag_embed(self.W_line)
        B_inv = torch.sum(hyper_graph.detach(), dim=-2)
        B_inv = torch.diag_embed(B_inv)
        softmax_w = torch.abs(self.W).detach()
        D_inv = torch.matmul(hyper_graph.detach(), softmax_w).sum(dim=-1)
        D_inv = torch.diag_embed(D_inv)
        D_inv = D_inv **(-0.5)
        B_inv = B_inv **(-1)
        D_inv[D_inv == float('inf')] = 0
        D_inv[D_inv == float('nan')] = 0
        B_inv[B_inv == float('inf')] = 0
        B_inv[B_inv == float('nan')] = 0
        A = torch.bmm(D_inv, hyper_graph)
        A = torch.matmul(A, torch.abs(self.W))
        A = torch.bmm(A, B_inv)
        A = torch.bmm(A, hyper_graph.transpose(-2, -1))
        A = torch.bmm(A, D_inv)
        X = torch.bmm(A, node_features)
        return X

class Encoder(nn.Module):
    def __init__(self, aggregator, feature_dim):
        super(Encoder, self).__init__()
        self.aggregator = aggregator
        self.feature_dim = feature_dim

    def forward(self, node_features, hyper_graph):
        output = self.aggregator.forward(node_features, hyper_graph)
        return output

class HGCNMixer(nn.Module):
    def __init__(self, args):
        super(HGCNMixer, self).__init__()
        self.args = args
        self.add_self = args.add_self
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.head_num = 1
        self.hyper_edge_num = args.hyper_edge_num
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.indiv_u_dim = int(np.prod(args.observation_shape))
        self.use_one_hot = False
        self.n_hyper_edge = self.hyper_edge_num
        if self.use_one_hot:
            self.n_hyper_edge += self.n_agents
        self.use_elu = True
        self.hyper_edge_net = nn.Sequential(
            nn.Linear(in_features=self.indiv_u_dim, out_features=self.hyper_edge_num),
            nn.ReLU(),
        )
        self.hidden_dim = 64
        self.encoder_1 = nn.ModuleList([Encoder(HGCN(self.n_hyper_edge, 1, self.hidden_dim, self.n_agents), self.indiv_u_dim) for _ in range(self.head_num)])
        self.encoder_2 = nn.ModuleList([Encoder(HGCN(self.n_hyper_edge, 1, self.hidden_dim, self.n_agents), self.indiv_u_dim) for _ in range(self.head_num)])
        self.hyper_weight_layer_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )
        self.hyper_const_layer_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )

        self.hyper_weight_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.n_agents)
        )
        self.hyper_const_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, 1)
        )

    def build_hyper_net(self, indiv_us):
        out = self.hyper_edge_net(indiv_us)
        mean = out.clone().detach().mean()
        out = out.reshape([out.shape[0], self.n_agents, -1])
        if self.use_one_hot:
            one_hot = torch.eye(self.n_agents)
            one_hot = one_hot.flatten().cuda()
            mean = out.clone().detach().mean()
            one_hot = one_hot * mean
            one_hot = one_hot.repeat(indiv_us.shape[0], 1).reshape([indiv_us.shape[0],self.n_agents, -1]).cuda()
            out = torch.cat([out, one_hot], dim=-1)
        return out.reshape([out.shape[0], out.shape[1], -1])

    def forward(self, agent_qs, states, indiv_us):
        bs = agent_qs.size(0)
        sl = agent_qs.size(1)
        agent_qs = agent_qs.view(-1, agent_qs.size(-1))
        indiv_us = indiv_us.reshape(-1, indiv_us.size(-2), indiv_us.size(-1))
        hyper_graph = self.build_hyper_net(indiv_us)
        states = states.reshape(-1, states.size(-1))
        hyper_graph = hyper_graph.reshape(-1, hyper_graph.size(-2), hyper_graph.size(-1))
        node_features = agent_qs.unsqueeze(dim=-1)
        # qs_tot = node_features.squeeze(dim=-1)
        qs_tot = self.encoder_2[0](self.encoder_1[0].forward(node_features, hyper_graph), hyper_graph).squeeze(dim=-1)
        hyper_weight_1 = torch.abs(self.hyper_weight_layer_1(states))
        hyper_const_1 = self.hyper_const_layer_1(states)
        q_tot = (qs_tot * hyper_weight_1) + hyper_const_1
        if self.use_elu:
            q_tot = F.elu(q_tot)
        hyper_weight = torch.abs(self.hyper_weight_layer(states))
        hyper_const = self.hyper_const_layer(states).squeeze(dim=-1)
        q_tot = (q_tot*hyper_weight).sum(dim=-1) + hyper_const.squeeze(dim=-1).squeeze(dim=-1)
        return q_tot.view(bs, sl, 1)
