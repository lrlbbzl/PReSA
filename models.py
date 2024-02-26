from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    head_vector: torch.tensor
    schema_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.addictive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.head_encoder = AutoModel.from_pretrained(args.pretrained_model)
        self.schema_encoder = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_encoder = AutoModel.from_pretrained(args.pretrained_model)

        self.hidden_size = 768
        self.linear = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.relu = nn.LeakyReLU()

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, schema_token_ids, schema_mask, schema_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        schema_vector = self._encode(self.schema_encoder,
                                 token_ids=schema_token_ids,
                                 mask=schema_mask,
                                 token_type_ids=schema_token_type_ids)

        tail_vector = self._encode(self.tail_encoder,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.head_encoder,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'schema_vector': schema_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        head, tail, schema = output_dict['head_vector'], output_dict['tail_vector'], output_dict['schema_vector']
        batch_size = head.size(0)
        labels = torch.arange(batch_size).to(head.device)

        target = self.relu(self.linear(torch.cat([head, schema, head - schema, head * schema], dim=1)))
        logits = torch.matmul(target, tail.t()) # (bs, bs)
        if self.training:
            logits -= torch.zeros_like(logits).fill_diagonal_(self.args.addictive_margin).to(logits.device)
        logits *= torch.exp(self.log_inv_t)

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.args.add_negatives:
            assert self.args.select_rank < batch_size and self.args.select_rank != 1
            schema_similarity = torch.matmul(schema, schema.t())
            idx = schema_similarity.topk(max((self.args.select_rank, )), 1, True, False).indices[:, -1]
            similar_schema = schema[idx]
            negative_target = self.relu(self.linear(torch.cat([head, similar_schema, head - similar_schema, head * similar_schema], dim=1)))
            negative_logits = torch.matmul(negative_target, tail.t())
            logits = torch.cat([logits, negative_logits], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'head_vector': head.detach(),
                'schema_vector': schema.detach(),
                'tail_vector': tail.detach()}

    def predict_target(self, output_dict: dict,):
        head, tail, schema = output_dict['head_vector'], output_dict['tail_vector'], output_dict['schema_vector']
        return self.relu(self.linear(torch.cat([head, schema, head - schema, head * schema], dim=1)))

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_encoder,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
