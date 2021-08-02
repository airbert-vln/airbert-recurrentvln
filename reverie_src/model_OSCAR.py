# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from param import args

from vlnbert.vlnbert_init import get_vlnbert_models


class VLNBERT(nn.Module):
    def __init__(self, directions=4, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitializing the VLN-BERT model ...')
        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = directions  # number of pano views as context

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(hidden_size+args.angle_feat_size, hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.obj_pos_encode = nn.Linear(5, args.angle_feat_size, bias=True)
        self.obj_projection = nn.Linear(feature_size+args.angle_feat_size, hidden_size, bias=True)
        self.obj_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, mode, sentence, token_type_ids=None, position_ids=None,
                attention_mask=None,
                action_feats=None, pano_feats=None, cand_feats=None, 
                obj_feats=None, obj_pos=None, already_dropfeat=False):
    
        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, sentence, position_ids=position_ids,
                        token_type_ids=token_type_ids, attention_mask=attention_mask)

            return encoded_sentence

        elif mode == 'visual':
            # attention mask: language_masks, pano_masks, candidate_masks, obj_masks
            state_action_embed = torch.cat((sentence[:,0,:], action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)
            state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:,1:,:]), dim=1)

            if not already_dropfeat:
                cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
                obj_feats[..., :-args.angle_feat_size] = self.drop_env(obj_feats[..., :-args.angle_feat_size])

            cand_feats_embed = self.img_projection(cand_feats)  # [2176 * 768] projection
            cand_feats_embed = self.cand_LayerNorm(cand_feats_embed)

            obj_feats_embed = self.obj_pos_encode(obj_pos)
            obj_feats_concat = torch.cat((obj_feats[..., :-args.angle_feat_size], obj_feats_embed, obj_feats[..., -args.angle_feat_size:]), dim=-1)
            obj_feats_embed = self.obj_projection(obj_feats_concat)
            obj_feats_embed = self.obj_LayerNorm(obj_feats_embed)

            cand_obj_feats_embed = torch.cat((cand_feats_embed, obj_feats_embed), dim=1)

            # logit is the attention scores over the candidate features
            h_t, logit, logit_obj = self.vln_bert(mode, state_feats,
                        attention_mask=attention_mask, img_feats=cand_obj_feats_embed)
            
            return h_t, logit, logit_obj

        else:
            ModuleNotFoundError


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
