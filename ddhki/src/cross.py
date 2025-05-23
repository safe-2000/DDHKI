import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, chnl=8):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = nn.ReLU()
        self.drop_layer = nn.Dropout(p=self.dropout) # Pytorch drop: ratio to zeroed
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        x = self.drop_layer(inputs)
        output = self.fc(x)
        return self.act(output)

class Bias(nn.Module):
    def __init__(self, dim):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.bias


class VectorCrossLayer(nn.Module):
    def __init__(self, dim):
        super(VectorCrossLayer, self).__init__()
        self.dim = dim
        self.W_vv = nn.Linear(dim, 1, bias=False)  # Renamed weights
        self.W_ev = nn.Linear(dim, 1, bias=False)
        self.W_ve = nn.Linear(dim, 1, bias=False)
        self.W_ee = nn.Linear(dim, 1, bias=False)

        self.bias_vertex = Bias(dim)  # Renamed bias
        self.bias_edge = Bias(dim)

    def forward(self, inputs):
        vertex, edge = inputs  # Renamed inputs

        # Reshape in a single line
        vertex_expanded = vertex.unsqueeze(2)
        edge_expanded = edge.unsqueeze(1)


        # Cross product calculation
        cross_product = torch.matmul(vertex_expanded, edge_expanded)
        transposed_cross_product = cross_product.permute(0, 2, 1)

        # Flatten in single lines
        flat_cross_product = cross_product.view(-1, self.dim)
        flat_transposed_cross_product = transposed_cross_product.contiguous().view(-1, self.dim)


        # Intermediate calculations, split into two lines each for vertex and edge
        vertex_int = self.W_vv(flat_cross_product)
        vertex_int += self.W_ev(flat_transposed_cross_product)

        edge_int = self.W_ve(flat_cross_product)
        edge_int += self.W_ee(flat_transposed_cross_product)


        # Reshape and apply bias
        vertex_int_reshaped = vertex_int.view(-1, self.dim)
        edge_int_reshaped = edge_int.view(-1, self.dim)

        vertex_out = self.bias_vertex(vertex_int_reshaped)
        edge_out = self.bias_edge(edge_int_reshaped)

        return vertex_out, edge_out