import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from cross import VectorCrossLayer
from diffusion_model import *


class DDHKI(nn.Module):
    def __init__(self, args, n_entity, n_relation, n_user, n_item):
        super(DDHKI, self).__init__()

        self.args = args
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg


        # int entity and relation 
        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        
        # attention 
        self.kg_attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )
        
        # senet 
        self.kg_senet = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.Sigmoid(),
                )


        # init
        self._init_weight()

        # cross layer
        self.user_kg_cross = VectorCrossLayer(self.dim)
        self.item_kg_cross = VectorCrossLayer(self.dim)


        # diffsuion
        noise_steps = 100
        beta_start = 0.0001
        beta_end = 0.001
        self.diffusion_model = DiffusionModel(self.dim, noise_steps, beta_start, beta_end)


    def get_diffusion_emb(self, x, is_Train=True):


        batch_size = x.shape[0]
        t = torch.randint(0, self.diffusion_model.noise_steps, (batch_size,))
        
        if is_Train:
            # forward
            x_t, noise = self.diffusion_model.forward_diffusion(x, t)
            
            # reverse
            x_t_denoised = self.diffusion_model.reverse_diffusion(x_t, t)

            return x_t_denoised
        else:
            weight = 0.5
            x_t_denoised = self.diffusion_model.reverse_diffusion(x, t)
            x_t_denoised = (1 - weight) * x + weight * x_t_denoised

            return x_t_denoised



    def get_user_full_emb(self, users, user_triple_set, is_Train=False):

        """ user 塔 """
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])


        # [batch_size, dim]
        user_emb_origin = user_emb_0.mean(dim=1)

     
        user_emb_origin_diff = self.get_diffusion_emb(user_emb_origin, is_Train)

        # diffusion loss
        diff_loss = F.mse_loss(user_emb_origin_diff, user_emb_origin)


        user_embeddings.append(user_emb_origin)


        # Hierarchical Knowledge Collaborative Layer and  Diffusion-based Collaborative Knowledge Denoising Layer
        user_knowledge_emb = None
        know_diff_loss = 0

        for i in range(self.n_layer): # n_layer * [batch_size, dim]
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i])

            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
       
            user_emb_i_diff = self.get_diffusion_emb(user_emb_i, is_Train)

            know_diff_loss += F.mse_loss(user_emb_i_diff, user_emb_i)


            if i == 0:
                user_knowledge_emb = h_emb.mean(dim=1)
            user_embeddings.append(user_emb_i)


        # Higher-order Knowledge Collaborative Interaction Layer
        if self.args.use_kg_cross:
            _, head_emb = self.user_kg_cross([user_emb_origin, user_knowledge_emb])
            user_embeddings.append(head_emb)


        # concat
        user_embeddings_concat = torch.cat(user_embeddings, axis=-1)

        return user_embeddings_concat, diff_loss + know_diff_loss


    def get_item_full_emb(self, items, item_triple_set, is_Train=False):

        """ item  """
        item_embeddings = []
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
  
        item_emb_origin_diff = self.get_diffusion_emb(item_emb_origin, is_Train)

        # diffusion loss
        diff_loss = F.mse_loss(item_emb_origin_diff, item_emb_origin)

        item_embeddings.append(item_emb_origin)

        # Hierarchical Knowledge Collaborative Layer and  Diffusion-based Collaborative Knowledge Denoising Layer
        item_knowledge_emb = None
        know_diff_loss = 0
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)

            item_emb_i_diff = self.get_diffusion_emb(item_emb_i, is_Train)
            item_embeddings.append(item_emb_i)

            know_diff_loss += F.mse_loss(item_emb_i_diff, item_emb_i)
        
            if i == 0:
                item_knowledge_emb = h_emb.mean(dim=1)

        # Higher-order Knowledge Collaborative Interaction Layer
        if self.args.use_kg_cross:
            _, head_emb = self.item_kg_cross([item_emb_origin, item_knowledge_emb])
            item_embeddings.append(head_emb)

        item_embeddings_concat = torch.cat(item_embeddings, axis=-1)

        return item_embeddings_concat, diff_loss + know_diff_loss


                
    def forward(
        self,
        users: torch.LongTensor,
        items: torch.LongTensor,
        user_triple_set: list,
        item_triple_set: list,
        epoch=0,
        is_Train=False
    ):       
        
        # user
        user_embeddings_concat, user_diff_loss = self.get_user_full_emb(users, user_triple_set, is_Train=is_Train)


        # item 
        item_embeddings_concat, item_diff_loss = self.get_item_full_emb(items, item_triple_set, is_Train=is_Train)


        # diff loss
        total_diff_loss = user_diff_loss + item_diff_loss


        # preduction
        scores = self.predict([user_embeddings_concat], [item_embeddings_concat])

        return scores, total_diff_loss
    


    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _parse_args(self, args, n_entity, n_relation, n_user, n_item):

        self.n_user = n_user
        self.n_item = n_item

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
        
 
    def _init_weight(self):
        
        def init_xavier_uniform(module):
            """ xavier_uniform"""
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

        # inut embedding 
        init_xavier_uniform(self.entity_emb)
        init_xavier_uniform(self.relation_emb)

        # init
        for layer in self.kg_attention:
            init_xavier_uniform(layer)



    def _knowledge_attention(self, h_emb, r_emb, t_emb):

        # [batch_size, triple_set_size]
        att_weights = self.kg_attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        att_weights_norm = F.softmax(att_weights,dim=-1)
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)

        if self.args.use_knowledge_gate:
            knowledge_gate_output = 2 * self.kg_senet(torch.cat((h_emb,r_emb),dim=-1))
            emb_i = knowledge_gate_output * emb_i

        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i