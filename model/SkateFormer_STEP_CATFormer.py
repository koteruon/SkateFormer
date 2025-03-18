import math
from typing import List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import (
    DropPath,
    Mlp,
    create_act_layer,
    create_conv2d,
    drop_path,
    get_norm_act_layer,
    trunc_normal_,
)
from torch import einsum

from utils import import_class

""" Partition and Reverse """


def type_1_partition(input, partition_size):  # partition_size = [N, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_1_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(
        B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1
    )
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_2_partition(input, partition_size):  # partition_size = [N, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, T // partition_size[0], partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 2, 5, 3, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_2_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(
        B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1
    )
    output = output.permute(0, 5, 1, 3, 4, 2).contiguous().view(B, -1, T, V)
    return output


def type_3_partition(input, partition_size):  # partition_size = [M, L]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], V // partition_size[1], partition_size[1])
    partitions = partitions.permute(0, 3, 4, 2, 5, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_3_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(
        B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1
    )
    output = output.permute(0, 5, 3, 1, 2, 4).contiguous().view(B, -1, T, V)
    return output


def type_4_partition(input, partition_size):  # partition_size = [M, K]
    B, C, T, V = input.shape
    partitions = input.view(B, C, partition_size[0], T // partition_size[0], partition_size[1], V // partition_size[1])
    partitions = partitions.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, partition_size[0], partition_size[1], C)
    return partitions


def type_4_reverse(partitions, original_size, partition_size):  # original_size = [T, V]
    T, V = original_size
    B = int(partitions.shape[0] / (T * V / partition_size[0] / partition_size[1]))
    output = partitions.view(
        B, T // partition_size[0], V // partition_size[1], partition_size[0], partition_size[1], -1
    )
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, -1, T, V)
    return output


""" 1D relative positional bias: B_{h}^{t} """


def get_relative_position_index_1d(T):
    coords = torch.stack(torch.meshgrid([torch.arange(T)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += T - 1
    return relative_coords.sum(-1)


""" MSA """


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, rel_type, num_heads=32, partition_size=(1, 1), attn_drop=0.0, rel=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.rel_type = rel_type
        self.num_heads = num_heads
        self.partition_size = partition_size
        self.scale = num_heads**-0.5
        self.attn_area = partition_size[0] * partition_size[1]
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.rel = rel

        if self.rel:
            if self.rel_type == "type_1" or self.rel_type == "type_3":
                self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * partition_size[0] - 1), num_heads))
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=0.02)
                self.ones = torch.ones(partition_size[1], partition_size[1], num_heads)
            elif self.rel_type == "type_2" or self.rel_type == "type_4":
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros((2 * partition_size[0] - 1), partition_size[1], partition_size[1], num_heads)
                )
                self.register_buffer("relative_position_index", get_relative_position_index_1d(partition_size[0]))
                trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _get_relative_positional_bias(self):
        if self.rel_type == "type_1" or self.rel_type == "type_3":
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], -1
            )
            relative_position_bias = (
                relative_position_bias.unsqueeze(1)
                .unsqueeze(3)
                .repeat(1, self.partition_size[1], 1, self.partition_size[1], 1, 1)
                .view(self.attn_area, self.attn_area, -1)
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)
        elif self.rel_type == "type_2" or self.rel_type == "type_4":
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.partition_size[0], self.partition_size[0], self.partition_size[1], self.partition_size[1], -1
            )
            relative_position_bias = (
                relative_position_bias.permute(0, 2, 1, 3, 4).contiguous().view(self.attn_area, self.attn_area, -1)
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        B_, N, C = input.shape
        qkv = input.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rel:
            attn = attn + self._get_relative_positional_bias()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        return output


""" SkateFormer Block """


class SkateFormerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_points=50,
        kernel_size=7,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(SkateFormerBlock, self).__init__()
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.type_3_size = type_3_size
        self.type_4_size = type_4_size
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.rel_type = ["type_1", "type_2", "type_3", "type_4"]

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        trunc_normal_(self.gconv, std=0.02)
        self.tconv = nn.Conv2d(
            in_channels // (2 * 2),
            in_channels // (2 * 2),
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads // (2 * 2),
        )

        # Attention layers
        attention = []
        for i in range(len(self.partition_function)):
            attention.append(
                MultiHeadSelfAttention(
                    in_channels=in_channels // (len(self.partition_function) * 2),
                    rel_type=self.rel_type[i],
                    num_heads=num_heads // (len(self.partition_function) * 2),
                    partition_size=self.partition_size[i],
                    attn_drop=attn_drop,
                    rel=rel,
                )
            )
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels, hidden_features=int(mlp_ratio * in_channels), act_layer=act_layer, drop=drop
        )

    def forward(self, input):
        B, C, T, V = input.shape

        # Partition
        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        # G-Conv
        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum("n c t u, v u -> n c t v", split_f_gconv[i], self.gconv[i])
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1))  # N C T V

        # T-Conv
        y.append(self.tconv(split_f_conv[1]))

        # Skate-MSA
        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)

        for i in range(len(self.partition_function)):
            C = split_f_attn[i].shape[1]
            input_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            input_partitioned = input_partitioned.view(-1, self.partition_size[i][0] * self.partition_size[i][1], C)
            y.append(self.reverse_function[i](self.attention[i](input_partitioned), (T, V), self.partition_size[i]))

        output = self.proj(torch.cat(y, dim=1).permute(0, 2, 3, 1).contiguous())
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)

        # Feed Forward
        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


""" Downsampling """


class PatchMergingTconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, stride=2, dilation=1):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.reduction = nn.Conv2d(
            dim_in, dim_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(dim_out)

    def forward(self, x):
        x = self.bn(self.reduction(x))
        return x


""" SkateFormer Block with Downsampling """


class SkateFormerBlockDS(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_points=50,
        kernel_size=7,
        downscale=False,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer_transformer=nn.LayerNorm,
    ):
        super(SkateFormerBlockDS, self).__init__()

        if downscale:
            self.downsample = PatchMergingTconv(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.downsample = None

        self.transformer = SkateFormerBlock(
            in_channels=out_channels,
            num_points=num_points,
            kernel_size=kernel_size,
            num_heads=num_heads,
            type_1_size=type_1_size,
            type_2_size=type_2_size,
            type_3_size=type_3_size,
            type_4_size=type_4_size,
            attn_drop=attn_drop,
            drop=drop,
            rel=rel,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer_transformer,
        )

    def forward(self, input):
        if self.downsample is not None:
            output = self.transformer(self.downsample(input))
        else:
            output = self.transformer(input)
        return output


""" SkateFormer Stage """


class SkateFormerStage(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        out_channels,
        first_depth=False,
        num_points=50,
        kernel_size=7,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer_transformer=nn.LayerNorm,
    ):
        super(SkateFormerStage, self).__init__()
        blocks = []
        for index in range(depth):
            blocks.append(
                SkateFormerBlockDS(
                    in_channels=in_channels if index == 0 else out_channels,
                    out_channels=out_channels,
                    num_points=num_points,
                    kernel_size=kernel_size,
                    downscale=((index == 0) & ~first_depth),
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path if isinstance(drop_path, float) else drop_path[index],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        output = input
        for block in self.blocks:
            output = block(output)
        return output


""" SkateFormer """


class SkateFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        depths=(2, 2, 2, 2),
        channels=(96, 192, 192, 192),
        num_classes=60,
        embed_dim=64,
        num_people=2,
        num_frames=64,
        num_points=50,
        kernel_size=7,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        attn_drop=0.0,
        head_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer_transformer=nn.LayerNorm,
        index_t=False,
        global_pool="avg",
        graph=None,
        head=["ViT-B/32"],
    ):

        super(SkateFormer, self).__init__()

        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        self.head_drop = head_drop
        self.index_t = index_t
        self.embed_dim = embed_dim

        if self.head_drop != 0:
            self.dropout = nn.Dropout(p=self.head_drop)
        else:
            self.dropout = None

        stem = []
        stem.append(
            nn.Conv2d(
                in_channels=in_channels, out_channels=2 * in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            )
        )
        stem.append(act_layer())
        stem.append(
            nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=3 * in_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
        )
        stem.append(act_layer())
        stem.append(
            nn.Conv2d(
                in_channels=3 * in_channels, out_channels=embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            )
        )
        self.stem = nn.ModuleList(stem)

        if self.index_t:
            self.joint_person_embedding = nn.Parameter(torch.zeros(embed_dim, num_points * num_people))
            trunc_normal_(self.joint_person_embedding, std=0.02)
        else:
            self.joint_person_temporal_embedding = nn.Parameter(
                torch.zeros(1, embed_dim, num_frames, num_points * num_people)
            )
            trunc_normal_(self.joint_person_temporal_embedding, std=0.02)

        # Init blocks
        drop_path = torch.linspace(0.0, drop_path, sum(depths)).tolist()
        stages = []
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            stages.append(
                SkateFormerStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    first_depth=index == 0,
                    num_points=num_points * num_people,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    type_1_size=type_1_size,
                    type_2_size=type_2_size,
                    type_3_size=type_3_size,
                    type_4_size=type_4_size,
                    attn_drop=attn_drop,
                    drop=drop,
                    rel=rel,
                    drop_path=drop_path[sum(depths[:index]) : sum(depths[: index + 1])],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer_transformer=norm_layer_transformer,
                )
            )
        self.stages = nn.ModuleList(stages)
        self.global_pool: str = global_pool
        self.fc = nn.Linear(channels[-1], num_classes)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_classes))

        # STEP CATFormer
        self.fc1 = nn.Linear(channels[-1], num_classes)
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2.0 / num_classes))
        self.head = head
        self.linear_head = nn.ModuleDict()
        self.linear_head["ViT-B/32"] = nn.Linear(channels[-1], 512)
        self.num_points = num_points
        # Graph
        Graph = import_class(graph)
        self.graph = Graph()
        A = self.graph.A
        self.A_vector = self.get_A(graph, k=1).float()

        # joint patition
        self.head_list = torch.Tensor(self.graph.transfer([2, 3])).long()
        self.hand_list = torch.Tensor(self.graph.transfer([4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24])).long()
        self.foot_list = torch.Tensor(self.graph.transfer([12, 13, 14, 15, 16, 17, 18, 19])).long()
        self.hip_list = torch.Tensor(self.graph.transfer([0, 1, 12, 16])).long()
        self.part_list = nn.ModuleList()
        for i in range(4):
            self.part_list.append(nn.Linear(channels[3], 512))

        # Temporal fusion
        self.fusion = [
            TemporalConv(channels[0], channels[1], kernel_size=1, stride=8, dilation=1),
            TemporalConv(channels[1], channels[2], kernel_size=1, stride=4, dilation=1),
            TemporalConv(channels[2], channels[3], kernel_size=1, stride=2, dilation=1),
            TemporalConv(channels[3], channels[3], kernel_size=1, stride=1, dilation=1),
        ]
        self.x_l = []
        self.fusion_last = nn.Conv2d(channels[3], channels[3], kernel_size=1)

        # Embedding
        self.to_joint_embedding = nn.Linear(channels[3], channels[3])
        self.pos_embedding = nn.Parameter(torch.randn(1, num_points, channels[3]))

        # Temporal
        self.Temporal_TransformerEncoder1 = Temporal_TransformerEncoderv2(
            num_points,
            channels[3],
            num_heads=8,
            ff_expand=4.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.Temporal_TransformerEncoder2 = Temporal_TransformerEncoderv2(
            num_points,
            channels[3],
            num_heads=8,
            ff_expand=4.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        # Spratial cross
        self.cross_hand = MultiScaleTransformerEncoder_hand(
            channels[3],
            channels[3],
            num_point=num_points,
            cross_attn_depth=1,
            cross_attn_heads=8,
            dropout=0.0,
            graph=self.graph,
        )
        self.cross_leg = MultiScaleTransformerEncoder_leg(
            channels[3],
            channels[3],
            num_point=num_points,
            cross_attn_depth=1,
            cross_attn_heads=8,
            dropout=0.0,
            graph=self.graph,
        )
        self.cross_hand_leg = MultiScaleTransformerEncoder_hand_leg(
            channels[3],
            channels[3],
            num_point=num_points,
            cross_attn_depth=1,
            cross_attn_heads=8,
            dropout=0.0,
            graph=self.graph,
        )
        self.cross_up_dowm = MultiScaleTransformerEncoder(
            channels[3],
            channels[3],
            num_point=num_points,
            cross_attn_depth=1,
            cross_attn_heads=8,
            dropout=0.0,
            graph=self.graph,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels[3] * num_points, channels[3] * num_points),
            nn.GELU(),
            nn.LayerNorm(channels[3] * num_points),
            nn.Linear(channels[3] * num_points, channels[3] * num_points),
            nn.GELU(),
            nn.LayerNorm(channels[3] * num_points),
        )
        self.norm = nn.LayerNorm(channels[3] * num_points)

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        if k == 0:
            return torch.from_numpy(I)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if "relative_position_bias_table" in n:
                nwd.add(n)
        return nwd

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes: int = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, input):
        output = input
        for index, stage in enumerate(self.stages):
            output = stage(output)
            x = self.fusion[index].to(output.device)(output)
            x = x.mean(-1, keepdim=True)
            self.x_l.append(x)
        return output

    def forward_head(self, input, pre_logits=False):
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        if self.dropout is not None:
            input = self.dropout(input)
        return input if pre_logits else self.head(input)

    def forward(self, input, index_t):
        B, C, T, V, M = input.shape

        output = input.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)  # [B, C, T, M * V]
        for layer in self.stem:
            output = layer(output)
        if self.index_t:
            te = torch.zeros(B, T, self.embed_dim).to(output.device)  # B, T, C
            div_term = torch.exp(
                (torch.arange(0, self.embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / self.embed_dim))
            ).to(output.device)
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum("b t c, c v -> b c t v", te, self.joint_person_embedding)
        else:
            output = output + self.joint_person_temporal_embedding
        output = self.forward_features(output)

        # Step catformer
        c_new = output.size(1)
        x = output.view(B, c_new, T // 8, M, V).permute(0, 3, 1, 2, 4).contiguous().view(B * M, c_new, T // 8, V)
        feature = x.view(B, M, c_new, T // 8, V)
        head_feature = self.part_list[0](feature[:, :, :, :, self.head_list].mean(4).mean(3).mean(1))
        hand_feature = self.part_list[1](feature[:, :, :, :, self.hand_list].mean(4).mean(3).mean(1))
        foot_feature = self.part_list[2](feature[:, :, :, :, self.foot_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[3](feature[:, :, :, :, self.hip_list].mean(4).mean(3).mean(1))

        x_lst = x.view(B, M, c_new, -1)
        x_lst = x_lst.mean(3).mean(1)
        feature_dict = dict()
        for name in self.head:
            feature_dict[name] = self.linear_head[name](x_lst)

        x_ab_ = x.view(B, M, c_new, -1)
        x_ab_ = x_ab_.mean(3).mean(1)
        output_ab = self.fc(x_ab_)

        # Temporal fusion concat
        p = torch.cat(self.x_l, -1)
        p = self.fusion_last(p)

        # Transformer
        B1, C1, T1, V1 = x.size()
        x_ = rearrange(x, "b c t v -> (b t) v c").contiguous()
        x_ = self.to_joint_embedding(x_)
        x_ += self.pos_embedding[:, : self.num_points]
        x_ = rearrange(x_, "(b t) v c -> b c t v", b=B1, t=T1).contiguous()

        x_s1 = self.cross_hand(x_, x_)  # hand
        x_s2 = self.cross_leg(x_, x_)  # leg
        x_t1 = self.Temporal_TransformerEncoder1(x_s1, x_s2, x_s1, p)  # q,k,v,p

        x_s1 = self.cross_hand_leg(x_s1, x_s2)  # hand leg
        x_s2 = self.cross_up_dowm(x_s2, x_s1)  # down up
        x_t2 = self.Temporal_TransformerEncoder2(x_s1, x_s2, x_s1, p)  # q,k,v,p

        x_t = x_t1 + x_t2

        x_t = rearrange(x_t, "b c f j -> b f (j c)")
        x_t = self.mlp(self.norm(x_t))
        x = rearrange(x_t, "b f (j c) -> b c f j", j=V1)

        # output = self.forward_head(output)
        return output


def SkateFormer_(**kwargs):
    return SkateFormer(depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), embed_dim=96, **kwargs)


# STEP CATFormer


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

        # initialize
        nn.init.normal_(self.to_k.weight, 0, math.sqrt(2.0 / inner_dim))
        nn.init.normal_(self.to_q.weight, 0, math.sqrt(2.0 / inner_dim))
        nn.init.normal_(self.to_v.weight, 0, math.sqrt(2.0 / inner_dim))
        self.apply(weights_init)

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)

        v = self.to_v(x_qkv)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Temporal_TransformerEncoderv2(nn.Module):
    def __init__(
        self,
        num_joint=25,
        dim_emb=128,
        num_heads=8,
        ff_expand=4.0,
        qkv_bias=False,
        attn_do_rate=0.0,
        proj_do_rate=0.0,
        drop_path=0.0,
    ):

        super(Temporal_TransformerEncoderv2, self).__init__()

        self.norm_q = nn.LayerNorm(dim_emb)
        self.norm_k = nn.LayerNorm(dim_emb)
        self.norm_v = nn.LayerNorm(dim_emb)
        self.attn = MDTA_T(dim_emb, 8)
        self.norm2 = nn.LayerNorm(dim_emb)
        self.ffn = GDFN(dim_emb, ff_expand)

        self.num_joints = num_joint
        self.add_coeff = nn.Parameter(torch.zeros(5, self.num_joints))

        self.transform = nn.Sequential(nn.BatchNorm2d(dim_emb), nn.GELU(), nn.Conv2d(dim_emb, dim_emb, kernel_size=1))

        self.bn = nn.BatchNorm2d(dim_emb)

    def forward(self, q, k, v, p, mask=None):
        b_o, c_o, t_o, v_o = q.shape

        q = torch.cat([q, p, q.mean(-1, keepdim=True)], -1)
        k = torch.cat([k, p, k.mean(-1, keepdim=True)], -1)
        v = torch.cat([v, p, v.mean(-1, keepdim=True)], -1)

        b, c, t, j = q.shape

        q_unnorm = q

        q = self.norm_q(q.reshape(b, c, -1).transpose(-2, -1).contiguous())
        k = self.norm_k(k.reshape(b, c, -1).transpose(-2, -1).contiguous())
        v = self.norm_v(v.reshape(b, c, -1).transpose(-2, -1).contiguous())

        q = q.transpose(-2, -1).contiguous().reshape(b, c, t, j)
        k = k.transpose(-2, -1).contiguous().reshape(b, c, t, j)
        v = v.transpose(-2, -1).contiguous().reshape(b, c, t, j)

        x = q_unnorm + self.attn(q, k, v)

        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, t, j)
        )

        local_feat = x[..., :v_o]
        global_feat = x[..., v_o:]

        global_feat = torch.einsum("nctd,dv->nctv", global_feat, self.add_coeff[:v_o])
        feat = local_feat + global_feat
        feat = self.transform(feat)
        x = self.bn(feat)

        return x


class MDTA_T(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA_T, self).__init__()
        self.num_heads = num_heads
        self.dim = channels / self.num_heads

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.v = Multi_DTemporalConv_Branch(channels, channels, kernel_size=7, stride=1, dilations=[2, 3])

        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # initialize
        self.apply(weights_init)

    def forward(self, q, k, v):
        b, c, t, j = v.shape

        q = self.q_conv(self.q(q))
        k = self.k_conv(self.k(k))
        v = self.v(v)

        q = rearrange(q, "b c t j  -> b t (j c)").contiguous()
        k = rearrange(k, "b c t j  -> b t (j c)").contiguous()
        v = rearrange(v, "b c t j  -> b t (j c)").contiguous()

        q = q.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, t, self.num_heads, -1).permute(0, 2, 1, 3)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        x = rearrange(x, "b t (j c) -> b c t j", j=j)
        out = self.project_out(x)
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)

        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class Multi_DTemporalConv_Branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=[1, 2, 3, 4], residual=True):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "# out channels should be multiples of # branches"
        global iii
        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        branch_channels = branch_channels
        branch_channels2 = branch_channels

        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    DTemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
                )
                for ks, dilation in zip(kernel_size, dilations)
            ]
        )

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels2, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                # nn.BatchNorm2d(branch_channels2)
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels2, kernel_size=1, padding=0, stride=(stride, 1)),
                # nn.BatchNorm2d(branch_channels2)
            )
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class DTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(DTemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=out_channels,
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class MultiScaleTransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_point=25,
        cross_attn_depth=1,
        cross_attn_heads=3,
        dropout=0.0,
        residual=True,
        att=False,
        graph=None,
    ):
        super().__init__()

        large_dim = (output_dim // 2) // 2
        small_dim = large_dim + (output_dim // 2)

        self.transformer_enc_small1 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small2 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(
            num_point,
            small_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.transformer_enc_large1 = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large2 = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large_T = Spatial_TransformerEncoder(
            num_point,
            large_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(small_dim, large_dim),
                        nn.Linear(large_dim, small_dim),
                        PreNorm(
                            large_dim, CrossAttention(large_dim, heads=8, dim_head=large_dim // 8, dropout=dropout)
                        ),
                        nn.Linear(large_dim, small_dim),
                        nn.Linear(small_dim, large_dim),
                        PreNorm(
                            small_dim, CrossAttention(small_dim, heads=8, dim_head=small_dim // 8, dropout=dropout)
                        ),
                    ]
                )
            )

        self.pool = "cls"

        self.mlp_head_small = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, output_dim))

        self.mlp_head_large = nn.Sequential(nn.LayerNorm(large_dim), nn.Linear(large_dim, output_dim))

        self.feedforward = FeedForward(
            in_features=output_dim * num_point,
            hidden_features=int(output_dim * num_point),
            out_features=output_dim * num_point,
            do_rate=0.0,
        )
        self.norm2 = nn.LayerNorm(output_dim * num_point)

        # initialize
        self.apply(weights_init)

        self.graph = graph

    def forward(self, x1, x2):

        up_list = torch.Tensor(self.graph.transfer([2, 3, 10, 11, 6, 7, 8, 9, 4, 5, 21, 22, 23, 24])).long()
        down_list = torch.Tensor(self.graph.transfer([16, 17, 18, 19, 12, 13, 14, 15, 0, 1])).long()

        headhand_feature1 = x1[:, :, :, up_list]
        foothip_feature1 = x1[:, :, :, down_list]

        headhand_feature2 = x2[:, :, :, up_list]
        foothip_feature2 = x2[:, :, :, down_list]

        foothip_feature1 = self.transformer_enc_small1(foothip_feature1)
        foothip_feature2 = self.transformer_enc_small2(foothip_feature2)

        headhand_feature1 = self.transformer_enc_large1(headhand_feature1)
        headhand_feature2 = self.transformer_enc_large2(headhand_feature2)

        xs = self.transformer_enc_small_T(foothip_feature1, foothip_feature2, foothip_feature1)
        xl = self.transformer_enc_large_T(headhand_feature1, headhand_feature2, headhand_feature1)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs = rearrange(xs, "b c t v -> (b t) v c")
        xl = rearrange(xl, "b c t v -> (b t) v c")

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim=1) if self.pool == "mean" else xs
        xl = xl.mean(dim=1) if self.pool == "mean" else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs = rearrange(xs, "(b t) v c -> b c t v", t=ts, v=vs)
        xl = rearrange(xl, "(b t) v c -> b c t v", t=tl, v=vl)

        branch_out = []
        branch_out.append(xs)
        branch_out.append(xl)
        x = torch.cat(branch_out, dim=3)

        b, c, f, j = x.shape
        x_tm = rearrange(x, "b c f j   -> b f (j c)")
        x_out = x_tm + self.feedforward(self.norm2(x_tm))
        x_out = rearrange(x_out, "b f (j c)  -> b c f j", j=j)
        return x_out


class MultiScaleTransformerEncoder_hand_leg(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_point=25,
        cross_attn_depth=1,
        cross_attn_heads=3,
        dropout=0.0,
        residual=True,
        att=False,
        graph=None,
    ):
        super().__init__()

        large_dim = (output_dim // 2) // 2
        small_dim = large_dim + (output_dim // 2)

        self.transformer_enc_small1 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small2 = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(
            num_point,
            small_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.transformer_enc_large1 = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large2 = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large_T = Spatial_TransformerEncoder(
            num_point,
            large_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(small_dim, large_dim),
                        nn.Linear(large_dim, small_dim),
                        PreNorm(
                            large_dim, CrossAttention(large_dim, heads=8, dim_head=large_dim // 8, dropout=dropout)
                        ),
                        nn.Linear(large_dim, small_dim),
                        nn.Linear(small_dim, large_dim),
                        PreNorm(
                            small_dim, CrossAttention(small_dim, heads=8, dim_head=small_dim // 8, dropout=dropout)
                        ),
                    ]
                )
            )

        self.pool = "cls"

        self.mlp_head_small = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, output_dim))

        self.mlp_head_large = nn.Sequential(nn.LayerNorm(large_dim), nn.Linear(large_dim, output_dim))

        self.feedforward = FeedForward(
            in_features=output_dim * num_point,
            hidden_features=int(output_dim * num_point),
            out_features=output_dim * num_point,
            do_rate=0.0,
        )
        self.norm2 = nn.LayerNorm(output_dim * num_point)

        # initialize
        self.apply(weights_init)

        self.graph = graph

    def forward(self, x1, x2):

        up_list = torch.Tensor(self.graph.transfer([2, 3, 10, 11, 6, 7, 8, 9, 4, 5, 21, 22, 23, 24])).long()
        down_list = torch.Tensor(self.graph.transfer([16, 17, 18, 19, 12, 13, 14, 15, 0, 1])).long()

        headhand_feature1 = x1[:, :, :, up_list]
        foothip_feature1 = x1[:, :, :, down_list]

        headhand_feature2 = x2[:, :, :, up_list]
        foothip_feature2 = x2[:, :, :, down_list]

        foothip_feature1 = self.transformer_enc_small1(foothip_feature1)
        foothip_feature2 = self.transformer_enc_small2(foothip_feature2)

        headhand_feature1 = self.transformer_enc_large1(headhand_feature1)
        headhand_feature2 = self.transformer_enc_large2(headhand_feature2)

        xs = self.transformer_enc_small_T(foothip_feature1, foothip_feature2, foothip_feature1)
        xl = self.transformer_enc_large_T(headhand_feature1, headhand_feature2, headhand_feature1)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs = rearrange(xs, "b c t v -> (b t) v c")
        xl = rearrange(xl, "b c t v -> (b t) v c")

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim=1) if self.pool == "mean" else xs
        xl = xl.mean(dim=1) if self.pool == "mean" else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs = rearrange(xs, "(b t) v c -> b c t v", t=ts, v=vs)
        xl = rearrange(xl, "(b t) v c -> b c t v", t=tl, v=vl)

        branch_out = []
        branch_out.append(xs)
        branch_out.append(xl)
        x = torch.cat(branch_out, dim=3)

        b, c, f, j = x.shape
        x_tm = rearrange(x, "b c f j   -> b f (j c)")
        x_out = x_tm + self.feedforward(self.norm2(x_tm))
        x_out = rearrange(x_out, "b f (j c)  -> b c f j", j=j)
        return x_out


class MultiScaleTransformerEncoder_hand(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_point=25,
        cross_attn_depth=1,
        cross_attn_heads=3,
        dropout=0.0,
        residual=True,
        att=False,
        graph=None,
    ):
        super().__init__()

        large_dim = (output_dim // 2) // 2
        small_dim = large_dim + (output_dim // 2)

        self.transformer_enc_small = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(
            num_point,
            small_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.transformer_enc_large = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large_T = Spatial_TransformerEncoder(
            num_point,
            large_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(small_dim, large_dim),
                        nn.Linear(large_dim, small_dim),
                        PreNorm(
                            large_dim, CrossAttention(large_dim, heads=8, dim_head=large_dim // 8, dropout=dropout)
                        ),
                        nn.Linear(large_dim, small_dim),
                        nn.Linear(small_dim, large_dim),
                        PreNorm(
                            small_dim, CrossAttention(small_dim, heads=8, dim_head=small_dim // 8, dropout=dropout)
                        ),
                    ]
                )
            )

        self.pool = "cls"

        self.mlp_head_small = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, output_dim))

        self.mlp_head_large = nn.Sequential(nn.LayerNorm(large_dim), nn.Linear(large_dim, output_dim))

        self.feedforward = FeedForward(
            in_features=output_dim * num_point,
            hidden_features=int(output_dim * num_point),
            out_features=output_dim * num_point,
            do_rate=0.0,
        )
        self.norm2 = nn.LayerNorm(output_dim * num_point)

        # initialize
        self.apply(weights_init)

        self.graph = graph

    def forward(self, x1, x2):

        up_list = torch.Tensor(self.graph.transfer([4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24])).long()
        down_list = torch.Tensor(self.graph.transfer([16, 17, 18, 19, 12, 13, 14, 15, 0, 1, 2, 3])).long()

        headhand_feature = x1[:, :, :, up_list]
        foothip_feature = x2[:, :, :, down_list]

        xs = self.transformer_enc_small(foothip_feature)
        xs = self.transformer_enc_small_T(xs, xs, xs)

        xl = self.transformer_enc_large(headhand_feature)
        xl = self.transformer_enc_large_T(xl, xl, xl)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs = rearrange(xs, "b c t v -> (b t) v c")
        xl = rearrange(xl, "b c t v -> (b t) v c")

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim=1) if self.pool == "mean" else xs
        xl = xl.mean(dim=1) if self.pool == "mean" else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs = rearrange(xs, "(b t) v c -> b c t v", t=ts)
        xl = rearrange(xl, "(b t) v c -> b c t v", t=tl)

        branch_out = []
        branch_out.append(xs)
        branch_out.append(xl)
        x = torch.cat(branch_out, dim=3)

        b, c, f, j = x.shape
        x_tm = rearrange(x, "b c f j   -> b f (j c)")
        x_out = x_tm + self.feedforward(self.norm2(x_tm))
        x_out = rearrange(x_out, "b f (j c)  -> b c f j", j=j)
        return x_out


class MultiScaleTransformerEncoder_leg(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        num_point=25,
        cross_attn_depth=1,
        cross_attn_heads=3,
        dropout=0.0,
        residual=True,
        att=False,
        graph=None,
    ):
        super().__init__()

        large_dim = (output_dim // 2) // 2
        small_dim = large_dim + (output_dim // 2)

        self.transformer_enc_small = nn.Conv2d(input_dim, small_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_small_T = Spatial_TransformerEncoder(
            num_point,
            small_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.transformer_enc_large = nn.Conv2d(input_dim, large_dim, kernel_size=1, padding=0, stride=(1, 1))
        self.transformer_enc_large_T = Spatial_TransformerEncoder(
            num_point,
            large_dim,
            num_heads=8,
            ff_expand=1.0,
            qkv_bias=False,
            attn_do_rate=0.0,
            proj_do_rate=0.0,
            drop_path=0.0,
        )

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(
                nn.ModuleList(
                    [
                        nn.Linear(small_dim, large_dim),
                        nn.Linear(large_dim, small_dim),
                        PreNorm(
                            large_dim, CrossAttention(large_dim, heads=8, dim_head=large_dim // 8, dropout=dropout)
                        ),
                        nn.Linear(large_dim, small_dim),
                        nn.Linear(small_dim, large_dim),
                        PreNorm(
                            small_dim, CrossAttention(small_dim, heads=8, dim_head=small_dim // 8, dropout=dropout)
                        ),
                    ]
                )
            )

        self.pool = "cls"

        self.mlp_head_small = nn.Sequential(nn.LayerNorm(small_dim), nn.Linear(small_dim, output_dim))

        self.mlp_head_large = nn.Sequential(nn.LayerNorm(large_dim), nn.Linear(large_dim, output_dim))

        self.feedforward = FeedForward(
            in_features=output_dim * num_point,
            hidden_features=int(output_dim * num_point),
            out_features=output_dim * num_point,
            do_rate=0.0,
        )
        self.norm2 = nn.LayerNorm(output_dim * num_point)

        # initialize
        self.apply(weights_init)

        self.graph = graph

    def forward(self, x1, x2):

        up_list = torch.Tensor(self.graph.transfer([4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24, 1, 2, 3])).long()
        down_list = torch.Tensor(self.graph.transfer([16, 17, 18, 19, 12, 13, 14, 15, 0])).long()

        headhand_feature = x1[:, :, :, up_list]
        foothip_feature = x2[:, :, :, down_list]

        xs = self.transformer_enc_small(foothip_feature)
        xs = self.transformer_enc_small_T(xs, xs, xs)

        xl = self.transformer_enc_large(headhand_feature)
        xl = self.transformer_enc_large_T(xl, xl, xl)

        bl, cl, tl, vl = xl.shape
        bs, cs, ts, vs = xs.shape

        xs = rearrange(xs, "b c t v -> (b t) v c")
        xl = rearrange(xl, "b c t v -> (b t) v c")

        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:

            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch
            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)

        xs = xs.mean(dim=1) if self.pool == "mean" else xs
        xl = xl.mean(dim=1) if self.pool == "mean" else xl

        xs = self.mlp_head_small(xs)
        xl = self.mlp_head_large(xl)

        xs = rearrange(xs, "(b t) v c -> b c t v", t=ts, v=vs)
        xl = rearrange(xl, "(b t) v c -> b c t v", t=tl, v=vl)

        branch_out = []
        branch_out.append(xs)
        branch_out.append(xl)
        x = torch.cat(branch_out, dim=3)

        b, c, f, j = x.shape
        x_tm = rearrange(x, "b c f j   -> b f (j c)")
        x_out = x_tm + self.feedforward(self.norm2(x_tm))
        x_out = rearrange(x_out, "b f (j c)  -> b c f j", j=j)
        return x_out


class Spatial_TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_joint=50,
        dim_emb=48,
        num_heads=8,
        ff_expand=1.0,
        qkv_bias=False,
        attn_do_rate=0.0,
        proj_do_rate=0.0,
        drop_path=0.0,
    ):
        super(Spatial_TransformerEncoder, self).__init__()
        self.normq_sp = nn.LayerNorm(dim_emb)
        self.normk_sp = nn.LayerNorm(dim_emb)
        self.normv_sp = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)

        self.feedforward = FeedForward(
            in_features=dim_emb, hidden_features=int(dim_emb * 4), out_features=dim_emb, do_rate=proj_do_rate
        )
        self.attention_sp = Attention(dim_emb, num_heads, qkv_bias, attn_do_rate, proj_do_rate)

    def forward(self, q, k, v, mask=None):

        b, c, f, j = q.shape

        q = rearrange(q, "b c f j   -> (b f) j c")
        k = rearrange(k, "b c f j   -> (b f) j c")
        v = rearrange(v, "b c f j   -> (b f) j c")

        ## spatial-MHA attention
        x_sp = q + self.attention_sp(self.normq_sp(q), self.normk_sp(k), self.normv_sp(v), mask=None)
        ## spatial-MHA ffn
        x_out = x_sp
        x_out = x_out + self.feedforward(self.norm2(x_out))
        x_out = rearrange(x_out, "(b f) j c  -> b c f j", b=b, f=f)
        return x_out


class Attention(nn.Module):
    def __init__(self, dim_emb, num_heads=8, qkv_bias=False, attn_do_rate=0.0, proj_do_rate=0.0):
        super().__init__()
        self.dim_emb = dim_emb
        self.num_heads = num_heads
        dim_each_head = dim_emb // num_heads
        self.scale = dim_each_head**-0.5

        self.W_q = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.W_k = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)
        self.W_v = nn.Linear(dim_emb, dim_emb, bias=qkv_bias)

        self.proj = nn.Linear(dim_emb, dim_emb)

        nn.init.normal_(self.W_q.weight, 0, math.sqrt(2.0 / dim_emb))
        nn.init.normal_(self.W_k.weight, 0, math.sqrt(2.0 / dim_emb))
        nn.init.normal_(self.W_v.weight, 0, math.sqrt(2.0 / dim_emb))

        nn.init.normal_(self.proj.weight, 0, math.sqrt(2.0 / dim_emb))

    def forward(self, q, k, v, mask=None):

        B, N, C = q.shape

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, do_rate=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2.0 / hidden_features))
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2.0 / out_features))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
