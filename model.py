import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from transformers import AutoModel, AutoTokenizer, DistilBertPreTrainedModel


class MLP(nn.Module):

    def __init__(self, hidden_size, num_labels, droupout_rate=0.5):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input):
        out = self.ffn(input)
        return out


class TEXTClassification(DistilBertPreTrainedModel):

    def __init__(self, config, num_labels=3):
        super(TEXTClassification, self).__init__(config)
        self.num_labels = num_labels

        self.bert = AutoModel.from_pretrained(config._name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.aspect_mlp = MLP(config.hidden_size, 3, 0.8)
        print(config.hidden_size)

        self.food_mlp = MLP(config.hidden_size, 2, 0.8)
        self.service_mlp = MLP(config.hidden_size, 2, 0.8)
        self.price_mlp = MLP(config.hidden_size, 2, 0.8)

    def forward(self, text, aspect=None, polarity=None, device=None):
        # text, aspect, polarity shape: (batch_size)

        encoded_input = self.tokenizer(text,
                                       return_tensors='pt',
                                       add_special_tokens=True,
                                       padding=True)
        device = aspect.device if aspect is not None else device
        encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}
        hidden_states = self.bert(**encoded_input)[0]
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        # aspect_res shape: (batch_size, 3)
        aspect_res = self.aspect_mlp(pooled_output)

        # food_res, price_res, service_res shape: (batch_size, 2)
        food_res = self.food_mlp(pooled_output)
        price_res = self.price_mlp(pooled_output)
        service_res = self.service_mlp(pooled_output)

        category = torch.argmax(aspect_res)
        if category == 0:
            polarity_res = food_res
        elif category == 1:
            polarity_res = price_res
        else:
            polarity_res = service_res

        # logist shape: (batch_size, 3, 2)
        logist = torch.stack([food_res, price_res, service_res], dim=1)

        if aspect is not None:
            loss_fct = nn.CrossEntropyLoss()
            aspect_loss = loss_fct(aspect_res.view(-1, 3), aspect.view(-1))
            polarity_loss = loss_fct(polarity_res.view(-1, 2), polarity.view(-1))

            loss = aspect_loss + polarity_loss
            return loss, aspect_res, polarity_res
        else:
            return aspect_res, logist


class EmbeddingLayer(nn.Module):

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config

        # define embedding layer
        self.userEmbLayer = nn.Embedding(config.max_user_num, config.hidden_size, 0)
        self.locEmbLayer = nn.Embedding(config.max_loc_num, config.hidden_size, 0)
        self.geoEmbLayer = nn.Embedding(config.max_geo_num, config.hidden_size, 0)
        self.cateEmbLayer = nn.Embedding(config.max_cate_num, config.hidden_size, 0)
        self.timeEmbLayer = nn.Embedding(config.max_time_num, config.hidden_size, 0)
        self.weekEmbLayer = nn.Embedding(config.max_week_num, config.hidden_size, 0)

        # init embedding layer
        nn.init.normal_(self.userEmbLayer.weight, std=0.1)
        nn.init.normal_(self.locEmbLayer.weight, std=0.1)
        nn.init.normal_(self.geoEmbLayer.weight, std=0.1)
        nn.init.normal_(self.cateEmbLayer.weight, std=0.1)
        nn.init.normal_(self.timeEmbLayer.weight, std=0.1)
        nn.init.normal_(self.weekEmbLayer.weight, std=0.1)

    def forward(self, user, traj, time, week, static_kg, loc_user_group, geo_user_group):
        # emb shape: (batch_size, max_sequence_length, hidden_size)
        user_emb = self.userEmbLayer(user)
        traj_emb = self.locEmbLayer(traj)
        time_emb = self.locEmbLayer(time)
        week_emb = self.locEmbLayer(week)

        static_kg = static_kg.clone()
        static_kg['user'].x = self.userEmbLayer(static_kg['user'].x)
        static_kg['location'].x = self.locEmbLayer(static_kg['location'].x)
        static_kg['area'].x = self.geoEmbLayer(static_kg['area'].x)
        if self.config.use_cate:
            static_kg['category'].x = self.cateEmbLayer(static_kg['category'].x)

        # loc_user_group shape: (batch_size, max_sequence_length, hidden_size)
        loc_user_group = self.userEmbLayer(loc_user_group).mean(2)
        geo_user_group = self.userEmbLayer(geo_user_group).mean(2)

        return user_emb, traj_emb, time_emb, week_emb, static_kg, loc_user_group, geo_user_group


class TKGEncoder(nn.Module):

    def __init__(self, metadata, hidden_size, num_heads, num_layers):
        super().__init__()
        self.agg_layer = AggregationLayer(
            metadata=metadata,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            return_user=False,
        )

    def forward(self, EmbeddingLayer, traj, tkg_dl, tkg_idx):
        # traj shape: (batch_size, max_sequence_length)
        # tkg_idx shape: (batch_size, max_sequence_length)

        # tkg_out shape: (tkg_length, max_loc_num, hidden_size)
        tkg_out = torch.Tensor([]).to(traj.device)
        for tkg in tkg_dl:

            tkg = tkg.to(traj.device)
            tkg['user'].x = EmbeddingLayer.userEmbLayer(tkg['user'].x)
            tkg['location'].x = EmbeddingLayer.locEmbLayer(tkg['location'].x)
            # out shape: (max_loc_num * tkg_batch_size, hidden_size)
            out = self.agg_layer(tkg.x_dict, tkg.edge_index_dict)
            # out shape: (tkg_batch_size, max_loc_num, hidden_size)
            out = out.view(-1, out.size(0) // tkg.num_graphs, out.size(1))
            tkg_out = torch.concat([tkg_out, out], dim=0)

        # tkg_traj shape: (batch_size, max_sequence_length, hidden_size)
        tkg_traj = self.tkg_filter(traj, tkg_out, tkg_idx)
        return tkg_traj, tkg_out

    def tkg_filter(self, traj, tkg_out, tkg_idx):
        # traj shape: (batch_size, max_sequence_length)
        # tkg_out shape: (tkg_length, max_loc_num, hidden_size)
        # tkg_idx shape: (batch_size, max_sequence_length)

        # selected_out shape: (tkg_length, batch_size, max_sequence_length, hidden_size)
        selected_out = tkg_out[:, traj]

        tkg_length = tkg_out.size(0)
        tkg_idx_list = torch.arange(tkg_length).to(traj.device)
        # tkg_mask shape: (tkg_length, batch_size, max_sequence_length, 1)
        tkg_mask = (tkg_idx_list.view(-1, 1, 1).repeat(1, traj.size(0), traj.size(1)) \
                        == tkg_idx.repeat(tkg_length, 1, 1)).long().unsqueeze(-1)

        # tkg_traj shape: (batch_size, max_sequence_length, hidden_size)
        tkg_traj = (selected_out * tkg_mask).sum(0)
        return tkg_traj


class StaticKGEncoder(nn.Module):

    def __init__(self, metadata, hidden_size, num_heads, num_layers) -> None:
        super().__init__()
        self.agg_layer = AggregationLayer(
            metadata=metadata,
            hidden_channels=hidden_size,
            out_channels=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            return_user=True,
        )

    def forward(self, static_kg):
        # static_kg_user shape: (max_user_num, hidden_size)
        # static_kg_loc shape: (max_loc_num, hidden_size)
        static_kg_user, static_kg_loc = self.agg_layer(static_kg.x_dict,
                                                       static_kg.edge_index_dict)
        return static_kg_user, static_kg_loc


class AggregationLayer(nn.Module):

    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels,
        num_heads,
        num_layers,
        return_user=False,
    ):
        super().__init__()
        self.return_user = return_user

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels,
                           hidden_channels,
                           metadata,
                           num_heads,
                           group='sum')
            self.convs.append(conv)

        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        tmp = x_dict['user'].clone()
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        if self.return_user:
            if x_dict['user'] is not None:
                return self.lin1(x_dict['user']), self.lin2(x_dict['location'])
            else:
                return tmp, self.lin2(x_dict['location'])
        return self.lin2(x_dict['location'])


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model,
                                               nhead,
                                               dropout=dropout,
                                               batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, q, k, v, attn_mask):
        x = v
        x = self.norm1(x + self._sa_block(q, k, v, attn_mask))
        x = x + self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, q, k, v, attn_mask):
        x = self.self_attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FusionLayer(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout) -> None:
        super().__init__()
        self.linear_absa = nn.Linear(6, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.trans_loc = TransformerEncoderLayer(hidden_size,
                                                 nhead=num_heads,
                                                 dropout=dropout)
        self.trans_user = TransformerEncoderLayer(hidden_size,
                                                  nhead=num_heads,
                                                  dropout=dropout)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, user_emb, traj_emb, tkg_traj, absa, static_kg_user, static_kg_loc,
                loc_user_group, geo_user_group):
        # user_emb shape: (batch_size, max_sequence_length, hidden_size)
        # traj_emb shape: (batch_size, max_sequence_length, hidden_size)
        # tkg_traj shape: (batch_size, max_sequence_length, hidden_size)
        # absa shape: (batch_size, max_sequence_length, 6)
        # static_kg_user shape: (batch_size, max_sequence_length, hidden_size)
        # static_kg_loc shape: (batch_size, max_sequence_length, hidden_size)
        # loc_user_group shape: (batch_size, max_sequence_length, hidden_size)
        # geo_user_group shape: (batch_size, max_sequence_length, hidden_size)

        # poi_fusion_output shape: (batch_size, max_sequence_length, hidden_size)
        poi_fusion_output = self.poi_fusion(traj_emb, tkg_traj, absa, static_kg_loc,
                                            loc_user_group)
        # user_fusion_ouput shape: (batch_size, max_sequence_length, hidden_size)
        user_fusion_ouput = self.user_fusion(user_emb, static_kg_user, loc_user_group,
                                             geo_user_group)
        return poi_fusion_output, user_fusion_ouput

    def poi_fusion(self, traj_emb, tkg_traj, absa, static_kg_loc, loc_user_group):
        # absa_query shape: (batch_size, hidden_size, max_sequence_length)
        if absa is not None:
            absa_query = self.linear_absa(absa).permute(0, 2, 1)

            # weight shape: (batch_size, max_sequence_length, max_sequence_length)
            weight1 = absa_query @ torch.tanh(self.linear1(traj_emb))
            weight2 = absa_query @ torch.tanh(self.linear2(tkg_traj))

            # weight shape: (batch_size)
            weight1 = torch.diagonal(weight1, dim1=-2, dim2=-1).mean(dim=1)
            weight2 = torch.diagonal(weight2, dim1=-2, dim2=-1).mean(dim=1)

            # beta shape: (batch_size)
            beta1 = torch.exp(weight1) / (torch.exp(weight1) + torch.exp(weight2))
            beta2 = torch.exp(weight2) / (torch.exp(weight1) + torch.exp(weight2))

            # output shape: (batch_size, max_sequence_length, hidden_size)
            output = beta1.unsqueeze(-1).unsqueeze(-1) * traj_emb + \
                            beta2.unsqueeze(-1).unsqueeze(-1) * tkg_traj
        else:
            output = traj_emb + tkg_traj

        mask_shape = [output.size(1), output.size(1)]
        attn_mask = torch.triu(torch.ones(mask_shape), 1).bool().to(output.device)
        # fusion_output shape: (batch_size, max_sequence_length, hidden_size)
        fusion_output = self.trans_loc(static_kg_loc,
                                       loc_user_group,
                                       output,
                                       attn_mask=attn_mask)
        return fusion_output

    def user_fusion(self, user_emb, static_kg_user, loc_user_group, geo_user_group):
        # user_group shape: (batch_size, max_sequence_length, hidden_size)
        user_group = torch.concat([loc_user_group, geo_user_group], dim=-1)
        user_group = self.linear(user_group)

        # fusion_output shape: (batch_size, max_sequence_length, hidden_size)
        fusion_output = self.trans_user(static_kg_user,
                                        user_group,
                                        user_emb,
                                        attn_mask=None)
        return fusion_output


class POIDecoder(nn.Module):

    def __init__(self, hidden_size) -> None:
        super().__init__()
        self.rnn = nn.LSTM(input_size=hidden_size * 2,
                           hidden_size=hidden_size * 2,
                           batch_first=True)
        self.dropout = nn.Dropout()

    def forward(self, poi_fusion_output, user_fusion_ouput, time_emb, week_emb):
        # poi_fusion_output shape: (batch_size, max_sequence_length, hidden_size)
        # user_fusion_ouput shape: (batch_size, max_sequence_length, hidden_size)

        # rnn_input shape: (batch_size, max_sequence_length, hidden_size * 2)
        rnn_input = torch.concat([poi_fusion_output, user_fusion_ouput], dim=-1)
        # rnn_input = self.dropout(rnn_input + torch.concat([time_emb, week_emb], dim=-1))

        # rnn_output shape: (batch_size, max_sequence_length, hidden_size * 2)
        rnn_output, _ = self.rnn(rnn_input)
        rnn_output = torch.relu(rnn_output) + rnn_input
        return rnn_output


class PoiModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.EmbeddingLayer = EmbeddingLayer(config)
        self.TKGEncoder = TKGEncoder(config.tkg_metadata,
                                     config.hidden_size,
                                     num_heads=1,
                                     num_layers=1)
        self.StaticKGEncoder = StaticKGEncoder(config.kg_metadata,
                                               config.hidden_size,
                                               num_heads=1,
                                               num_layers=1)
        self.FusionLayer = FusionLayer(config.hidden_size, num_heads=1, dropout=0.1)
        self.POIDecoder = POIDecoder(config.hidden_size)
        self.mlp = MLP(config.hidden_size * 2, config.max_loc_num)

    def static_kg_loss(self, tkg_out, static_kg_loc):
        # tkg_out shape: (tkg_length, max_loc_num, hidden_size)
        # static_kg_loc shape: (max_loc_num, hidden_size)

        loss_static = torch.zeros(1).to(tkg_out.device)

        for time_step in range(tkg_out.size(0)):
            step = (self.config.step * math.pi / 180) * (time_step + 1)
            sim_matrix = F.cosine_similarity(static_kg_loc, tkg_out[time_step])
            mask = (math.cos(step) - sim_matrix) > 0
            stamp = torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            loss_static += stamp / tkg_out.size(1)
        return loss_static

    def forward(self, user, traj, time, week, absa, tkg_dl, static_kg, tkg_idx,
                loc_user_group, geo_user_group):
        # user/traj/geo shape: (batch_size, max_sequence_length)
        # absa shape: (batch_size, max_sequence_length, 6)
        # tkg_idx shape: (batch_size, max_sequence_length)
        # loc_user_group shape: (batch_size, max_sequence_length, loc_group_num)
        # geo_user_group shape: (batch_size, max_sequence_length, geo_group_num)

        # shape: (batch_size, max_sequence_length, hidden_size)
        user_emb, traj_emb, time_emb, week_emb, static_kg, loc_user_group, geo_user_group = self.EmbeddingLayer(
            user, traj, time, week, static_kg, loc_user_group, geo_user_group)

        # tkg_traj shape: (batch_size, max_sequence_length, hidden_size)
        tkg_traj, tkg_out = self.TKGEncoder(self.EmbeddingLayer, traj, tkg_dl, tkg_idx)

        # static_kg_user shape: (max_user_num, hidden_size)
        # static_kg_loc shape: (max_loc_num, hidden_size)
        static_kg_user, static_kg_loc = self.StaticKGEncoder(static_kg)
        loss_static = self.static_kg_loss(tkg_out, static_kg_loc)

        # static_kg_user shape: (batch_size, max_sequence_length, hidden_size)
        # static_kg_loc shape: (batch_size, max_sequence_length, hidden_size)
        static_kg_user = static_kg_user[user]
        static_kg_loc = static_kg_loc[traj]

        # poi_fusion_output shape: (batch_size, max_sequence_length, hidden_size)
        # user_fusion_ouput shape: (batch_size, max_sequence_length, hidden_size)
        poi_fusion_output, user_fusion_ouput = self.FusionLayer(
            user_emb, traj_emb, tkg_traj, absa, static_kg_user, static_kg_loc,
            loc_user_group, geo_user_group)

        # decoder_output shape: (batch_size, max_sequence_length, hidden_size * 2)
        decoder_output = self.POIDecoder(poi_fusion_output, user_fusion_ouput, time_emb,
                                         week_emb)

        # pred shape: (batch_size, max_sequence_length, max_loc_num)
        pred = self.mlp(decoder_output)

        return pred, loss_static
