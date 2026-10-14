from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data import DRUG_MODELS, PROTEIN_MODELS

MODEL_NAME = "MolProtX"
A9V2_MODEL_NAME = "MolProtX_A9v2"
A9V2_ARCHITECTURE = "a9_variable_views_v2"


class NormalizedPositionalEncoding(nn.Module):
    """在统一后的特征坐标上加入 ProtSATT 风格的归一化位置信号。"""

    def __init__(self, dimension: int, mode: str, scale_init: float) -> None:
        super().__init__()
        if mode not in {"normalized", "none"}:
            raise ValueError(f"不支持的位置编码模式：{mode}")
        self.mode = mode
        position = torch.linspace(0.0, 1.0, steps=dimension, dtype=torch.float32)
        self.register_buffer("position", position, persistent=True)
        self.scale = nn.Parameter(
            torch.tensor(float(scale_init), dtype=torch.float32),
            requires_grad=mode != "none",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return features
        return features + self.scale.to(features.dtype) * self.position.to(features.dtype)


class InputProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.float())


class ProtSATTVectorSelfAttention(nn.Module):
    """ProtSATT 风格：注意力后展平，并投影为新的固定长度特征。"""

    def __init__(
        self,
        token_dim: int,
        sequence_length: int,
        output_flat_dim: int,
        num_heads: int,
        dropout: float,
        attention_weighting: str = "softmax",
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim 必须能被 num_heads 整除。")
        self.token_dim = token_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        if attention_weighting not in {"silu", "softmax"}:
            raise ValueError("attention_weighting 必须是 silu 或 softmax。")
        self.attention_weighting = attention_weighting
        self.query = nn.Linear(token_dim, token_dim, bias=False)
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.layer_norm = nn.LayerNorm(token_dim, eps=1e-12)
        self.flatten_projection = nn.Linear(sequence_length * token_dim, output_flat_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return inputs.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1:] != (self.sequence_length, self.token_dim):
            raise ValueError(
                f"SelfAttention 输入形状应为 [B,{self.sequence_length},{self.token_dim}]，"
                f"实际为 {tuple(inputs.shape)}"
            )
        query = self._split_heads(self.query(inputs))
        key = self._split_heads(self.key(inputs))
        value = self._split_heads(self.value(inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = (
            torch.softmax(scores, dim=-1)
            if self.attention_weighting == "softmax"
            else F.silu(scores)
        )
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        context = context.view(inputs.shape[0], self.sequence_length, self.token_dim)
        context = self.layer_norm(context + inputs)
        flattened = context.flatten(start_dim=1)
        return F.silu(self.dropout(self.flatten_projection(flattened)))


class ProtSATTDirectedCrossAttention(nn.Module):
    """同一模态内两个 PLM 视角之间的有向特征融合。"""

    def __init__(
        self,
        token_dim: int,
        sequence_length: int,
        output_dim: int,
        num_heads: int,
        dropout: float,
        residual_coef: float,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim 必须能被 num_heads 整除。")
        self.token_dim = token_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.query = nn.Linear(token_dim, token_dim, bias=False)
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.layer_norm = nn.LayerNorm(token_dim, eps=1e-12)
        self.output = nn.Linear(sequence_length * token_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_coef = float(residual_coef)

    def _split_heads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, _ = inputs.shape
        return inputs.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query_inputs: torch.Tensor, key_value_inputs: torch.Tensor) -> torch.Tensor:
        expected = (self.sequence_length, self.token_dim)
        if query_inputs.shape[1:] != expected or key_value_inputs.shape[1:] != expected:
            raise ValueError("模态内 Cross-Attention 的两个输入形状必须一致。")
        query = self._split_heads(self.query(query_inputs))
        key = self._split_heads(self.key(key_value_inputs))
        value = self._split_heads(self.value(key_value_inputs))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous()
        context = context.view(query_inputs.shape[0], self.sequence_length, self.token_dim)
        # 与 ProtSATT 原实现一致：融合残差来自提供 K/V 的目标视角，系数为 0.5。
        context = self.layer_norm(context + self.residual_coef * key_value_inputs)
        return F.silu(self.dropout(self.output(context.flatten(start_dim=1))))


class ProtSATTBranch(nn.Module):
    """Active PLM views only; each ordered pair has one directed fusion layer."""

    DIRECTED_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

    def __init__(
        self,
        model_names: tuple[str, ...],
        input_dims: dict[str, int],
        branch_config: dict[str, Any],
        relation_dim: int,
        positional_mode: str,
        positional_scale_init: float,
    ) -> None:
        super().__init__()
        if len(model_names) < 2 or len(set(model_names)) != len(model_names):
            raise ValueError("分支需要至少两个不重复的 embedding 视角。")
        self.model_names = tuple(model_names)
        self.directed_pairs = tuple((i, j) for i in range(len(model_names))
                                    for j in range(len(model_names)) if i != j)
        self.relation_names = tuple((self.model_names[i], self.model_names[j])
                                    for i, j in self.directed_pairs)
        projected_dim = int(branch_config["projected_dim"])
        first_token_dim = int(branch_config["first_token_dim"])
        first_output_dim = int(branch_config["first_output_dim"])
        deep_token_dim = int(branch_config["deep_token_dim"])
        deep_output_dim = int(branch_config["deep_output_dim"])
        fusion_token_dim = int(branch_config["fusion_token_dim"])
        num_heads = int(branch_config["attention_heads"])
        dropout = float(branch_config["dropout"])
        fusion_residual_coef = float(branch_config.get("fusion_residual_coef", 0.5))
        attention_weighting = str(branch_config.get("attention_weighting", "softmax"))

        if projected_dim % first_token_dim:
            raise ValueError("projected_dim 必须能被 first_token_dim 整除。")
        if first_output_dim % deep_token_dim:
            raise ValueError("first_output_dim 必须能被 deep_token_dim 整除。")
        if deep_output_dim % fusion_token_dim:
            raise ValueError("deep_output_dim 必须能被 fusion_token_dim 整除。")

        first_sequence_length = projected_dim // first_token_dim
        deep_sequence_length = first_output_dim // deep_token_dim
        fusion_sequence_length = deep_output_dim // fusion_token_dim
        self.first_shape = (first_sequence_length, first_token_dim)
        self.deep_shape = (deep_sequence_length, deep_token_dim)
        self.fusion_shape = (fusion_sequence_length, fusion_token_dim)

        self.input_projections = nn.ModuleDict(
            {
                name: InputProjection(input_dims[name], projected_dim, dropout)
                for name in self.model_names
            }
        )
        self.positional_encodings = nn.ModuleDict(
            {
                name: NormalizedPositionalEncoding(projected_dim, positional_mode, positional_scale_init)
                for name in self.model_names
            }
        )
        self.first_attention = nn.ModuleDict(
            {
                name: ProtSATTVectorSelfAttention(
                    token_dim=first_token_dim,
                    sequence_length=first_sequence_length,
                    output_flat_dim=first_output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_weighting=attention_weighting,
                )
                for name in self.model_names
            }
        )
        self.deep_attention = nn.ModuleDict(
            {
                name: ProtSATTVectorSelfAttention(
                    token_dim=deep_token_dim,
                    sequence_length=deep_sequence_length,
                    output_flat_dim=deep_output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_weighting=attention_weighting,
                )
                for name in self.model_names
            }
        )
        self.fusion_layers = nn.ModuleList(
            [
                ProtSATTDirectedCrossAttention(
                    token_dim=fusion_token_dim,
                    sequence_length=fusion_sequence_length,
                    output_dim=relation_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    residual_coef=fusion_residual_coef,
                )
                for _ in self.directed_pairs
            ]
        )
        self.relation_embedding = nn.Parameter(torch.zeros(len(self.directed_pairs), relation_dim))
        nn.init.normal_(self.relation_embedding, mean=0.0, std=0.02)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        views: list[torch.Tensor] = []
        for name in self.model_names:
            projected = self.input_projections[name](features[name])
            projected = self.positional_encodings[name](projected)
            first_input = projected.view(projected.shape[0], *self.first_shape)
            first_output = self.first_attention[name](first_input)
            deep_input = first_output.view(first_output.shape[0], *self.deep_shape)
            deep_output = self.deep_attention[name](deep_input)
            views.append(deep_output.view(deep_output.shape[0], *self.fusion_shape))

        relations = [
            layer(views[source], views[target])
            for layer, (source, target) in zip(self.fusion_layers, self.directed_pairs, strict=True)
        ]
        stacked = torch.stack(relations, dim=1)
        return stacked + self.relation_embedding.unsqueeze(0).to(stacked.dtype)


class BidirectionalCrossModalLayer(nn.Module):
    def __init__(self, dimension: int, num_heads: int, feedforward_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.protein_queries_drug = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.drug_queries_protein = nn.MultiheadAttention(
            dimension, num_heads, dropout=dropout, batch_first=True
        )
        self.protein_attention_norm = nn.LayerNorm(dimension)
        self.drug_attention_norm = nn.LayerNorm(dimension)
        hidden = dimension * feedforward_multiplier
        self.protein_ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dimension)
        )
        self.drug_ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dimension)
        )
        self.protein_ffn_norm = nn.LayerNorm(dimension)
        self.drug_ffn_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        protein_context, _ = self.protein_queries_drug(protein, drug, drug, need_weights=False)
        drug_context, _ = self.drug_queries_protein(drug, protein, protein, need_weights=False)
        protein_updated = self.protein_attention_norm(protein + self.dropout(protein_context))
        drug_updated = self.drug_attention_norm(drug + self.dropout(drug_context))
        protein_updated = self.protein_ffn_norm(
            protein_updated + self.dropout(self.protein_ffn(protein_updated))
        )
        drug_updated = self.drug_ffn_norm(drug_updated + self.dropout(self.drug_ffn(drug_updated)))
        return protein_updated, drug_updated


class AttentionPooling(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Linear(dimension, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(inputs).squeeze(-1), dim=-1)
        return torch.sum(inputs * weights.unsqueeze(-1), dim=1)


class ExplicitPairInteraction(nn.Module):
    """保留模态内信息，并显式建模匹配、差异和低秩双线性交互。"""

    def __init__(
        self,
        dimension: int,
        bilinear_rank: int,
        bilinear_output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dimension <= 0 or bilinear_rank <= 0 or bilinear_output_dim <= 0:
            raise ValueError("显式交互的 dimension、bilinear_rank 和输出维度必须大于 0。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("显式交互 dropout 必须位于 [0, 1)。")

        self.protein_pool = AttentionPooling(dimension)
        self.drug_pool = AttentionPooling(dimension)
        self.protein_projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
        )
        self.drug_projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
        )
        self.protein_factor = nn.Linear(dimension, bilinear_rank, bias=False)
        self.drug_factor = nn.Linear(dimension, bilinear_rank, bias=False)
        self.bilinear_output = nn.Sequential(
            nn.LayerNorm(bilinear_rank),
            nn.Linear(bilinear_rank, bilinear_output_dim),
            nn.GELU(),
        )
        self.output_dim = dimension * 4 + bilinear_output_dim
        self.output_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        # Each side is pooled independently; structural deletion may leave 2 vs 6 tokens.
        if (protein.ndim != 3 or drug.ndim != 3
                or protein.shape[0] != drug.shape[0] or protein.shape[2] != drug.shape[2]
                or protein.shape[1] == 0 or drug.shape[1] == 0):
            raise ValueError("显式交互要求两侧 batch 和特征维度一致，且均有非空 relation tokens。")
        protein_vector = self.protein_projection(self.protein_pool(protein))
        drug_vector = self.drug_projection(self.drug_pool(drug))
        bilinear = self.protein_factor(protein_vector) * self.drug_factor(drug_vector)
        bilinear = self.bilinear_output(bilinear)
        features = torch.cat(
            (
                protein_vector,
                drug_vector,
                protein_vector * drug_vector,
                torch.abs(protein_vector - drug_vector),
                bilinear,
            ),
            dim=-1,
        )
        return self.dropout(self.output_norm(features))


class MolProtXModel(nn.Module):
    """Original Full: paired cross-modal pooling plus explicit interaction."""
    def __init__(self, input_dims: dict[str, dict[str, int]], config: dict[str, Any]) -> None:
        super().__init__()
        relation_dim = int(config["relation_dim"])
        positional_mode = str(config.get("positional_encoding", "none"))
        positional_scale_init = float(config.get("positional_scale_init", 1.0))
        self.protein_branch = ProtSATTBranch(
            model_names=PROTEIN_MODELS,
            input_dims=input_dims["protein"],
            branch_config=config["protein_branch"],
            relation_dim=relation_dim,
            positional_mode=positional_mode,
            positional_scale_init=positional_scale_init,
        )
        self.drug_branch = ProtSATTBranch(
            model_names=DRUG_MODELS,
            input_dims=input_dims["drug"],
            branch_config=config["drug_branch"],
            relation_dim=relation_dim,
            positional_mode=positional_mode,
            positional_scale_init=positional_scale_init,
        )
        cross_config = config["cross_modal"]
        self.cross_modal_layers = nn.ModuleList(
            [
                BidirectionalCrossModalLayer(
                    dimension=relation_dim,
                    num_heads=int(cross_config["num_heads"]),
                    feedforward_multiplier=int(cross_config["feedforward_multiplier"]),
                    dropout=float(cross_config["dropout"]),
                )
                for _ in range(int(cross_config["num_layers"]))
            ]
        )
        fused_dim = relation_dim * 2
        interaction_config = config["explicit_interaction"]
        self.explicit_interaction = ExplicitPairInteraction(
            dimension=relation_dim,
            bilinear_rank=int(interaction_config["bilinear_rank"]),
            bilinear_output_dim=int(interaction_config["bilinear_output_dim"]),
            dropout=float(interaction_config["dropout"]),
        )
        classifier_config = config["classifier"]
        self.pooling = AttentionPooling(fused_dim)
        classifier_input_dim = fused_dim + self.explicit_interaction.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, int(classifier_config["hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(classifier_config["dropout"])),
            nn.Linear(int(classifier_config["hidden_dim"]), int(classifier_config["bottleneck_dim"])),
            nn.GELU(),
            nn.Dropout(float(classifier_config["dropout"])),
            nn.Linear(int(classifier_config["bottleneck_dim"]), 1),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        protein: dict[str, torch.Tensor],
        drug: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        protein_relations = self.protein_branch(protein)
        drug_relations = self.drug_branch(drug)
        explicit_features = self.explicit_interaction(protein_relations, drug_relations)
        for layer in self.cross_modal_layers:
            protein_relations, drug_relations = layer(protein_relations, drug_relations)
        fused = torch.cat((protein_relations, drug_relations), dim=-1)
        cross_modal_features = self.pooling(fused)
        classifier_features = torch.cat((cross_modal_features, explicit_features), dim=-1)
        return self.classifier(classifier_features).squeeze(-1)


class A9V2Model(nn.Module):
    """A9 core with independent pooling, supporting unequal relation counts."""

    def __init__(self, input_dims: dict[str, dict[str, int]], config: dict[str, Any]) -> None:
        super().__init__()
        self.architecture = A9V2_ARCHITECTURE
        relation_dim = int(config["relation_dim"])
        active = config.get("active_models", {"protein": PROTEIN_MODELS, "drug": DRUG_MODELS})
        for modality, allowed in (("protein", PROTEIN_MODELS), ("drug", DRUG_MODELS)):
            names = active[modality]
            if len(names) < 2 or len(set(names)) != len(names) or not set(names) <= set(allowed):
                raise ValueError(f"{modality} 必须选择至少两个不重复的合法视角。")
        common = {"relation_dim": relation_dim,
                  "positional_mode": str(config.get("positional_encoding", "none")),
                  "positional_scale_init": float(config.get("positional_scale_init", 1.0))}
        self.protein_branch = ProtSATTBranch(tuple(active["protein"]), input_dims["protein"],
                                             config["protein_branch"], **common)
        self.drug_branch = ProtSATTBranch(tuple(active["drug"]), input_dims["drug"],
                                          config["drug_branch"], **common)
        cross = config["cross_modal"]
        self.cross_modal_layers = nn.ModuleList([
            BidirectionalCrossModalLayer(relation_dim, int(cross["num_heads"]),
                                         int(cross["feedforward_multiplier"]), float(cross["dropout"]))
            for _ in range(int(cross["num_layers"]))
        ])
        self.protein_pool = AttentionPooling(relation_dim)
        self.drug_pool = AttentionPooling(relation_dim)
        self.classifier = self.make_classifier(2 * relation_dim, config["classifier"])
        self.apply(self._initialize_weights)

    _initialize_weights = staticmethod(MolProtXModel._initialize_weights)

    @staticmethod
    def make_classifier(input_dim: int, config: dict[str, Any]) -> nn.Sequential:
        hidden, bottleneck = int(config["hidden_dim"]), int(config["bottleneck_dim"])
        dropout = float(config["dropout"])
        return nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck), nn.GELU(), nn.Dropout(dropout), nn.Linear(bottleneck, 1),
        )

    def cross_modal_features(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        for layer in self.cross_modal_layers:
            protein, drug = layer(protein, drug)
        return torch.cat((self.protein_pool(protein), self.drug_pool(drug)), dim=-1)

    def forward(self, protein: dict[str, torch.Tensor], drug: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.cross_modal_features(self.protein_branch(protein), self.drug_branch(drug))
        return self.classifier(features).squeeze(-1)


# Keep the old import name available for tools reading historical checkpoints.
LegacyMolProtXModel = MolProtXModel


class LegacyA9Model(MolProtXModel):
    """The historical A9 used paired concatenation before pooling."""

    def __init__(self, input_dims: dict, config: dict) -> None:
        super().__init__(input_dims, config)
        self.explicit_interaction = None
        dimension = 2 * int(config["relation_dim"])
        self.classifier[0] = nn.LayerNorm(dimension)
        self.classifier[1] = nn.Linear(dimension, int(config["classifier"]["hidden_dim"]))
        self.classifier[1].apply(self._initialize_weights)

    def forward(self, protein: dict, drug: dict) -> torch.Tensor:
        protein, drug = self.protein_branch(protein), self.drug_branch(drug)
        for layer in self.cross_modal_layers:
            protein, drug = layer(protein, drug)
        return self.classifier(self.pooling(torch.cat((protein, drug), dim=-1))).squeeze(-1)


def training_model_class(config: dict) -> type[nn.Module]:
    """The default is the original Full; A9v2 requires its explicit saved config."""
    architecture = config.get("architecture")
    if architecture == A9V2_ARCHITECTURE:
        return A9V2Model
    if architecture is None and "explicit_interaction" in config and not config.get("active_models"):
        return MolProtXModel
    raise ValueError("主训练配置应为原 Full，或显式指定的 A9v2 配置；不能混用结构。")


def model_from_checkpoint(input_dims: dict, config: dict, state: dict) -> nn.Module:
    """Load Full, historical A9 and A9v2 weights strictly, without dropping parameters."""
    if config.get("architecture") == A9V2_ARCHITECTURE:
        model = A9V2Model(input_dims, config)
    elif any(key.startswith("explicit_interaction.") for key in state):
        model = MolProtXModel(input_dims, config)
    elif "pooling.score.weight" in state:
        model = LegacyA9Model(input_dims, config)
    else:
        raise ValueError("无法识别权重的模型版本；不能将历史权重当作新版主模型加载。")
    model.load_state_dict(state, strict=True)
    return model
