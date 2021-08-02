# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
import torch.nn.functional as F
from param import args

from vlnbert.vlnbert_init import get_vlnbert_models

class VLNBERT(nn.Module):
    def __init__(self, directions=4, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitializing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = directions  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        v_hidden_size = self.vln_bert.config.v_hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(v_hidden_size+args.angle_feat_size, v_hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)

    def forward(self, mode, sentence, token_type_ids=None, position_ids=None, 
                lang_masks=None, action_feats=None, pano_feats=None, 
                cand_feats=None, cand_masks=None, 
                obj_feats=None, obj_pos=None, obj_masks=None,
                h_t=None, already_dropfeat=False, act_t=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, lang_masks=lang_masks)
            return init_state, encoded_sentence

        elif mode == 'visual':
            # attention_mask: [lang_mask, cand_mask, obj_mask]
            state_action_embed = torch.cat((h_t, action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            obj_feats[..., :-4] = self.drop_env(obj_feats[..., :-4])
            
            # logit is the attention scores over the candidate features
            h_t, logit, logit_obj = \
                self.vln_bert(mode, sentence, cand_feats=cand_feats, 
                obj_feats=obj_feats, obj_pos=obj_pos, act_t=act_t,
                lang_masks=lang_masks, cand_masks=cand_masks, obj_masks=obj_masks,
                state_embeds=state_with_action)

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
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
