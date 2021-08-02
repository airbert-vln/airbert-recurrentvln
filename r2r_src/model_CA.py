# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

from vlnbert.vlnbert_init import get_vlnbert_models

class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitializing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT

        hidden_size = self.vln_bert.config.hidden_size
        v_hidden_size = self.vln_bert.config.v_hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(v_hidden_size+args.angle_feat_size, v_hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)

        self.visn2state_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)
        self.lang2state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.state_proj = nn.Linear(hidden_size+v_hidden_size*2, v_hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)

    def forward(self, mode, sentence, token_type_ids=None, position_ids=None, 
                lang_mask=None, attention_mask=None, action_feats=None, pano_feats=None, 
                cand_feats=None, cand_mask=None, h_t=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, lang_mask=lang_mask)

            return init_state, encoded_sentence

        elif mode == 'visual':

            state_action_embed = torch.cat((h_t, action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            
            # logit is the attention scores over the candidate features
            h_t, logit, attended_language, attended_visual, language_attention_probs, visual_attention_probs = \
                self.vln_bert(mode, sentence, lang_mask=lang_mask, 
                    cand_mask=cand_mask, cand_feats=cand_feats,
                    state_embeds=state_with_action)

            # update agent's state, unify history, language and vision
            state_output = torch.cat(
                (h_t, self.visn2state_LayerNorm(attended_visual), self.lang2state_LayerNorm(attended_language)), 
                dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            return state_proj, logit, language_attention_probs, visual_attention_probs

        else:
            ModuleNotFoundError


class ObjectVLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(ObjectVLNBERT, self).__init__()
        print('\nInitializing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT

        hidden_size = self.vln_bert.config.hidden_size
        v_hidden_size = self.vln_bert.config.v_hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(v_hidden_size+args.angle_feat_size, v_hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)

        self.drop_env = nn.Dropout(p=args.featdropout)

        self.visn2state_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)
        self.lang2state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.state_proj = nn.Linear(hidden_size+v_hidden_size*2, v_hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(v_hidden_size, eps=layer_norm_eps)

    def forward(self, mode, sentence, token_type_ids=None, position_ids=None, lang_mask=None, 
                cand_feats=None, cand_mask=None, obj_feats=None, obj_pos=None, obj_mask=None,
                action_feats=None, h_t=None, act_t=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, lang_mask=lang_mask)

            return init_state, encoded_sentence

        elif mode == 'visual':

            state_action_embed = torch.cat((h_t, action_feats), 1)
            state_with_action = self.action_state_project(state_action_embed)
            state_with_action = self.action_LayerNorm(state_with_action)

            cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            obj_feats = self.drop_env(obj_feats)

            # logit is the attention scores over the candidate features
            h_t, logit, attended_language, attended_visual, language_attention_probs, visual_attention_probs = \
                self.vln_bert(mode, sentence, token_type_ids=token_type_ids, lang_mask=lang_mask, 
                    cand_feats=cand_feats, cand_mask=cand_mask, 
                    obj_feats=obj_feats, obj_pos=obj_pos, obj_mask=obj_mask,
                    state_embeds=state_with_action, act_t=act_t,
                    obj_in_logits=args.obj_in_logits)

            # update agent's state, unify history, language and vision
            state_output = torch.cat(
                (h_t, self.visn2state_LayerNorm(attended_visual), self.lang2state_LayerNorm(attended_language)), 
                dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)

            # h_t, logit, language_attention_probs, visual_attention_probs = \
            #     self.vln_bert(mode, sentence, token_type_ids=token_type_ids, lang_mask=lang_mask, 
            #         cand_feats=cand_feats, cand_mask=cand_mask, 
            #         obj_feats=obj_feats, obj_pos=obj_pos, obj_mask=obj_mask,
            #         state_embeds=state_with_action, act_t=act_t)
                    
            # state_proj = h_t

            return state_proj, logit, language_attention_probs, visual_attention_probs

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
