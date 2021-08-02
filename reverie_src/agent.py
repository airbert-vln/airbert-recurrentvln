import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
import utils
from utils import padding_idx, add_idx, Tokenizer, print_progress
from param import args
from collections import defaultdict

import model_OSCAR, model_CA


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v, 'predObjId': r} for k, (v,r) in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v, 'predObjId': r} for k, (v,r) in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = (traj['path'], traj['predObjId'])
                if looped:
                    break


class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        if args.vlnbert == 'oscar':
            self.vln_bert = model_OSCAR.VLNBERT(directions=args.directions,
                feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_OSCAR.Critic().cuda()
        elif args.vlnbert == 'vilbert':
            self.vln_bert = model_CA.VLNBERT(
                feature_size=self.feature_size + args.angle_feat_size).cuda()
            self.critic = model_CA.Critic().cuda()

        # Optimizers
        self.vln_bert_optimizer = args.optimizer(self.vln_bert.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.criterion_REF = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        # self.ndtw_criterion = utils.ndtw_initialize()
        self.objProposals, self.obj2viewpoint = utils.loadObjProposals()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)

    def _sort_batch(self, obs, sorted_instr=True):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        if sorted_instr:
            seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
            sorted_tensor = seq_tensor[perm_idx]
            perm_idx = list(perm_idx)
        else:
            sorted_tensor = seq_tensor
            perm_idx = None

        mask = (sorted_tensor != padding_idx)    # seq_lengths[0] is the Maximum length

        token_type_ids = torch.zeros_like(mask)

        visual_mask = torch.ones(args.directions).bool()
        visual_mask = visual_mask.unsqueeze(0).repeat(mask.size(0),1)
        visual_mask = torch.cat((mask, visual_mask), -1)

        return sorted_tensor.long().cuda(), \
               mask.bool().cuda(),  token_type_ids.long().cuda(), \
               visual_mask.bool().cuda(), \
               list(seq_lengths), perm_idx

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.directions, self.feature_size + args.angle_feat_size), dtype=np.float32)

        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat

        return torch.from_numpy(features).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) for ob in obs]
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = cc['feature']

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def _object_variable(self, obs):
        cand_obj_leng = [len(ob['candidate_obj'][2]) + 1 for ob in obs] # +1 is for no REF
        if args.vlnbert == 'vilbert':
            cand_obj_feat = np.zeros((len(obs), max(cand_obj_leng), self.feature_size + 4), dtype=np.float32)
        elif args.vlnbert == 'oscar':
            cand_obj_feat = np.zeros((len(obs), max(cand_obj_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        cand_obj_pos = np.zeros((len(obs), max(cand_obj_leng), 5), dtype=np.float32)

        for i, ob in enumerate(obs):
            obj_local_pos, obj_features, candidate_objId = ob['candidate_obj']
            for j, cc in enumerate(candidate_objId):
                cand_obj_feat[i, j, :] = obj_features[j]
                cand_obj_pos[i, j, :] = obj_local_pos[j]

        return torch.from_numpy(cand_obj_feat).cuda(), torch.from_numpy(cand_obj_pos).cuda(), cand_obj_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        # f_t = self._feature_variable(obs)      # Image features from obs
        f_t = None
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        obj_feat, obj_pos, obj_leng = self._object_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng

    def _teacher_action(self, obs, ended, cand_size):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = cand_size - 1
        return torch.from_numpy(a).cuda()

    def _teacher_REF(self, obs, just_ended):
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if not just_ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                candidate_objs = ob['candidate_obj'][2]
                for k, kid in enumerate(candidate_objs):
                    if kid == ob['objId']:
                        a[i] = k
                        break
                else:
                    a[i] = args.ignoreid
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])

        if perm_idx is None:
            perm_idx = range(len(perm_obs))

        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

                state = self.env.env.sims[idx].getState()
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset: # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        # Reorder the language input for the encoder (do not ruin the original code)
        sentence, language_attention_mask, token_type_ids, \
            visual_attention_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'token_type_ids': token_type_ids}
        # (batch_size, seq_len, hidden_size)
        if args.vlnbert == 'oscar':
            language_inputs['attention_mask'] = language_attention_mask
            language_features = self.vln_bert(**language_inputs)
        elif args.vlnbert == 'vilbert':
            language_inputs['lang_masks'] = language_attention_mask
            h_t, language_features = self.vln_bert(**language_inputs)
            language_attention_mask = language_attention_mask[:, 1:]

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
            'predObjId': None
        } for ob in perm_obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        # last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            # last_ndtw[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env
        just_ended = np.array([False] * batch_size)

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        stop_mask = torch.tensor([False] * batch_size).cuda().unsqueeze(1)
        entropys = []
        ml_loss = 0.
        ref_loss = 0.

        # For test result submission: no backtracking
        visited = [set() for _ in range(batch_size)]

        for t in range(self.episode_len):

            input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)

            # the first [CLS] token, initialized by the language BERT, servers
            # as the agent's state passing through time steps
            if args.vlnbert != 'vilbert' and t >= 1:
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)
            
            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            'obj_feats':          obj_feat,
                            'obj_pos':            obj_pos,
                            'already_dropfeat':   (speaker is not None)}
            if args.vlnbert == 'oscar':
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)
                visual_inputs['attention_mask'] = visual_attention_mask
            elif args.vlnbert == 'vilbert':
                visual_inputs.update({
                    'h_t': h_t,
                    'lang_masks': language_attention_mask,
                    'cand_masks': visual_temp_mask,
                    'obj_masks': obj_temp_mask,
                    'act_t': t,
                })
            h_t, logit, logit_REF = self.vln_bert(**visual_inputs)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            candidate_mask = torch.cat((candidate_mask, stop_mask), dim=-1)
            logit.masked_fill_(candidate_mask, -float('inf'))
                
            candidate_mask_obj = utils.length2mask(obj_leng)
            logit_REF.masked_fill_(candidate_mask_obj, -float('inf'))

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(perm_obs, ended, candidate_mask.size(1))
                ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if ((next_id == visual_temp_mask.size(1)) or (t == self.episode_len-1)) and (not ended[i]):  # just stopped and forced stopped
                    just_ended[i] = True
                    if self.feedback == 'argmax':
                        _, ref_t = logit_REF[i].max(0)
                        if ref_t != obj_leng[i]-1:  # decide not to do REF
                            traj[i]['predObjId'] = perm_obs[i]['candidate_obj'][2][ref_t]

                    if args.submit:
                        if obj_leng[i] == 1:
                            traj[i]['predObjId'] = int("0")
                        else:
                            _, ref_t = logit_REF[i][:obj_leng[i]-1].max(0)
                            try:
                                traj[i]['predObjId'] = int(perm_obs[i]['candidate_obj'][2][ref_t])
                            except:
                                import pdb; pdb.set_trace()
                else:
                    just_ended[i] = False

                if (next_id == visual_temp_mask.size(1)) or (next_id == args.ignoreid) or (ended[i]):    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            ''' Supervised training for REF '''
            if train_ml is not None:
                target_obj = self._teacher_REF(perm_obs, just_ended)
                ref_loss += self.criterion_REF(logit_REF, target_obj)

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                # ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(perm_obs):
                    dist[i] = ob['distance']
                    # path_act = [vp[0] for vp in traj[i]['path']]
                    # ndtw_score[i] = self.ndtw_criterion[ob['scan']](path_act, ob['gt_path'], metric='ndtw')
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:                              # If the action now is end
                            # navigation success if the target object is visible when STOP
                            # end_viewpoint_id = ob['scan'] + '_' + ob['viewpoint']
                            # if self.objProposals.__contains__(end_viewpoint_id):
                            #     if ob['objId'] in self.objProposals[end_viewpoint_id]['objId']:
                            #         reward[i] = 2.0 + ndtw_score[i] * 2.0
                            #     else:
                            #         reward[i] = -2.0
                            # else:
                            #     reward[i] = -2.0
                            if dist[i] < 1.0:                             # Correct
                                reward[i] = 2.0  # + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Change of distance and nDTW reward
                            reward[i] = - (dist[i] - last_dist[i])
                            # ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0  # + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0  # + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                # last_ndtw[:] = ndtw_score

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            input_a_t, f_t, candidate_feat, candidate_leng, obj_feat, obj_pos, obj_leng = self.get_input_feat(perm_obs)
            if args.vlnbert != 'vilbert':
                language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1)

            visual_temp_mask = (utils.length2mask(candidate_leng) == 0).bool()
            obj_temp_mask = (utils.length2mask(obj_leng) == 0).bool()
            visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)

            self.vln_bert.vln_bert.config.directions = max(candidate_leng)
            self.vln_bert.vln_bert.config.obj_directions = max(obj_leng)
            ''' Visual BERT '''
            visual_inputs = {'mode':              'visual',
                            'sentence':           language_features,
                            'token_type_ids':     token_type_ids,
                            'action_feats':       input_a_t,
                            'pano_feats':         f_t,
                            'cand_feats':         candidate_feat,
                            'obj_feats':          obj_feat,
                            'obj_pos':            obj_pos,
                            'already_dropfeat':   (speaker is not None)}
            if args.vlnbert == 'oscar':
                visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask, obj_temp_mask), dim=-1)
                visual_inputs['attention_mask'] = visual_attention_mask
            elif args.vlnbert == 'vilbert':
                visual_inputs.update({
                    'h_t': h_t,
                    'lang_masks': language_attention_mask,
                    'cand_masks': visual_temp_mask,
                    'obj_masks': obj_temp_mask,
                    'act_t': len(hidden_states),
                })
            last_h_, _, _ = self.vln_bert(**visual_inputs)

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item())

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())
            self.loss += ref_loss * args.ref_loss_weight / batch_size
            self.logs['REF_loss'].append(ref_loss.item() * args.ref_loss_weight / batch_size)

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        # import pdb; pdb.set_trace()

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample': # agents in IL and RL separately
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()

            if args.aug is None:
                print_progress(iter, n_iters, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1

    def load_pretrain(self, path):
        ''' Loads parameters from pretrained network '''
        load_states = torch.load(path)
        # print(self.vln_bert.state_dict()['candidate_att_layer.linear_in.weight'])
        # print(self.vln_bert.state_dict()['visual_bert.bert.encoder.layer.9.intermediate.dense.weight'])

        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(load_states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS FOUND IN MODEL")
                for ikey in model_keys:
                    if ikey not in load_keys:
                        print('key not in model: ', ikey)
                for ikey in load_keys:
                    if ikey not in model_keys:
                        print('key not in loaded states: ', ikey)

            state.update(load_states[name]['state_dict'])
            model.load_state_dict(state)

        all_tuple = [("vln_bert", self.vln_bert)]
        for param in all_tuple:
            recover_state(*param)

        return load_states['vln_bert']['epoch'] - 1