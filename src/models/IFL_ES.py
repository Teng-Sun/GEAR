"""
From: https://github.com/thuiar/Self-MM
Paper: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis
"""
# self supervised multimodal multi-task learning network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import math

from .subNets import BertTextEncoder

__all__ = ['SELF_MM']

class GeneralizedCELoss(nn.Module):

    def __init__(self, args, q):
        super(GeneralizedCELoss, self).__init__()
        self.args = args
        self.q = q
        self.criterion = nn.L1Loss(reduction='none')
             
    def forward(self, logits, targets):
        loss_l1 = self.criterion(logits.detach(), targets)
        if self.args.gce == 'tanh':
            loss = torch.tanh((1/loss_l1) ** self.q) 
        elif self.args.gce == 'sigmoid':
            loss = 2 * torch.sigmoid(-loss_l1)
        elif self.args.gce == 'test1':
            loss =  torch.exp(-loss_l1 / self.q)
        elif self.args.gce == 'test2':
            loss = 1 / (1 + self.q * self.q * loss_l1 * loss_l1) 
        else:
            loss = 0
        loss = loss * self.criterion(logits, targets)
        return loss 

class text_model_base(nn.Module):
    def __init__(self, args):
        super(text_model_base, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        self.model_text = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers, pretrained=args.pretrained)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=args.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(args.text_out, args.swap_dim)

        self.feature = nn.Sequential(
            nn.Linear(args.swap_dim * 2, args.swap_dim),
            nn.ReLU(),
        )
        self.final = nn.Linear(args.swap_dim, 1)

    def forward(self, text):

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze(1).int().detach().cpu()
        text = self.model_text(text)[:,0,:]

        # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)

        return text_h


class text_model(nn.Module):
    def __init__(self, args):
        super(text_model, self).__init__()

        self.args = args
        self.model_c = text_model_base(args)
        self.model_b = text_model_base(args)
        self.criterion = nn.L1Loss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(args, q=args.q)

    def forward(self, text, labels, epoch=0):
        batch_scores_c = self.model_c(text)
        batch_scores_b = self.model_b(text)
        z_c = torch.cat((batch_scores_c, batch_scores_b.detach()), dim=1)
        z_b = torch.cat((batch_scores_c.detach(), batch_scores_b), dim=1)
        batch_feature_c = self.model_c.feature(z_c)
        batch_feature_b = self.model_b.feature(z_b)
        batch_scores_b1 = self.model_b.final(batch_feature_b)
        
        # loss_c = self.criterion(batch_scores_c1, labels).detach()
        loss_b = self.criterion(batch_scores_b1, labels).detach()
        
        # loss_weight = loss_b / (loss_b + loss_c + 1e-8)
        # loss_dis_conflict = self.criterion(batch_scores_c1, labels) * loss_weight.to(self.args.device)
        #print("******************", loss_dis_conflict, loss_weight)
        loss_dis_align = self.bias_criterion(batch_scores_b1, labels)
        
        if epoch > self.args.swap_epochs:
            indices = np.random.permutation(batch_scores_b.size(0))
            z_b_swap = batch_scores_b[indices]         # z tilde
            label_swap = labels[indices]     # y tilde
            # Prediction using z_swap=[z_l, z_b tilde]
            # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_mix_conflict = torch.cat((batch_scores_c, z_b_swap.detach()), dim=1)
            z_mix_align = torch.cat((batch_scores_c.detach(), z_b_swap), dim=1)
            # Prediction using z_swap
            pred_mix_conflict = self.model_c.feature(z_mix_conflict)
            #pred_mix_conflict = self.model_c.final(pred_mix_conflict)
            pred_mix_align = self.model_b.feature(z_mix_align)
            pred_mix_align = self.model_b.final(pred_mix_align)
            # loss_swap_conflict = self.criterion(pred_mix_conflict, labels) * loss_weight.to(self.args.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
            loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
            lambda_swap = self.args.lambda_swap                                         # Eq.3 lambda_swap_b
        else:
            # before feature-level augmentation
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda_swap = 0
            pred_mix_conflict = 0

        # loss_swap = loss_swap_conflict.mean() + args.lambda_dis*loss_swap_align.mean()
        # loss_dis = loss_dis_conflict.mean() + args.lambda_dis*loss_dis_align.mean()
        # loss = loss_dis + lambda_swap * loss_swap #+ 1e-10*independ_loss #+ 0.00001*l1_loss
        loss = lambda_swap * self.args.lambda_dis * loss_swap_align.mean() + self.args.lambda_dis * loss_dis_align.mean()

        return loss, loss_b, batch_feature_c, pred_mix_conflict

class audio_model_base(nn.Module):
    def __init__(self, args):
        super(audio_model_base, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        audio_in, video_in = args.feature_dims[1:]
        self.model_audio = AuViSubNet(audio_in, args.a_lstm_hidden_size, args.audio_out, \
                            num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        
        # the classify layer for text
        self.post_audio_dropout = nn.Dropout(p=args.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(args.audio_out, args.swap_dim)

        self.feature = nn.Sequential(
            nn.Linear(args.swap_dim * 2, args.swap_dim),
            nn.ReLU(),
        )
        self.final = nn.Linear(args.swap_dim, 1)

    def forward(self, audio, audio_lengths):

        audio = self.model_audio(audio, audio_lengths)

        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)

        return audio_h


class audio_model(nn.Module):
    def __init__(self, args):
        super(audio_model, self).__init__()
        self.args = args
        
        self.model_c = audio_model_base(args)
        self.model_b = audio_model_base(args)
        self.criterion = nn.L1Loss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(args, q=args.q)

    def forward(self, audio, auido_lengths, labels, epoch=0):
        batch_scores_c = self.model_c(audio, auido_lengths)
        batch_scores_b = self.model_b(audio, auido_lengths)
        z_c = torch.cat((batch_scores_c, batch_scores_b.detach()), dim=1)
        z_b = torch.cat((batch_scores_c.detach(), batch_scores_b), dim=1)
        batch_feature_c = self.model_c.feature(z_c)
        batch_feature_b = self.model_b.feature(z_b)
        # batch_scores_c1 = self.model_c.final(batch_feature_c)
        batch_scores_b1 = self.model_b.final(batch_feature_b)
        
        # loss_c = self.criterion(batch_scores_c1, labels).detach()
        loss_b = self.criterion(batch_scores_b1, labels).detach()
        
        # loss_weight = loss_b / (loss_b + loss_c + 1e-8)
        # loss_dis_conflict = self.criterion(batch_scores_c1, labels) * loss_weight.to(self.args.device)
        #print("******************", loss_dis_conflict, loss_weight)
        loss_dis_align = self.bias_criterion(batch_scores_b1, labels)
        if epoch > self.args.swap_epochs:
            indices = np.random.permutation(batch_scores_b.size(0))
            z_b_swap = batch_scores_b[indices]         # z tilde
            label_swap = labels[indices]     # y tilde
            # Prediction using z_swap=[z_l, z_b tilde]
            # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_mix_conflict = torch.cat((batch_scores_c, z_b_swap.detach()), dim=1)
            z_mix_align = torch.cat((batch_scores_c.detach(), z_b_swap), dim=1)
            # Prediction using z_swap
            pred_mix_conflict = self.model_c.feature(z_mix_conflict)
            #pred_mix_conflict = self.model_c.final(pred_mix_conflict)
            pred_mix_align = self.model_b.feature(z_mix_align)
            pred_mix_align = self.model_b.final(pred_mix_align)
            # loss_swap_conflict = self.criterion(pred_mix_conflict, labels) * loss_weight.to(self.args.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
            loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
            lambda_swap = self.args.lambda_swap                                         # Eq.3 lambda_swap_b
        else:
            # before feature-level augmentation
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda_swap = 0
            pred_mix_conflict = 0

        # loss_swap = loss_swap_conflict.mean() + args.lambda_dis*loss_swap_align.mean()
        # loss_dis = loss_dis_conflict.mean() + args.lambda_dis*loss_dis_align.mean()
        # loss = loss_dis + lambda_swap * loss_swap #+ 1e-10*independ_loss #+ 0.00001*l1_loss
        loss = lambda_swap * self.args.lambda_dis*loss_swap_align.mean() + self.args.lambda_dis*loss_dis_align.mean()

        return loss, loss_b, batch_feature_c, pred_mix_conflict

class video_model_base(nn.Module):
    def __init__(self, args):
        super(video_model_base, self).__init__()
        # text subnets
        self.aligned = args.need_data_aligned
        audio_in, video_in = args.feature_dims[1:]
        self.model_video = AuViSubNet(video_in, args.v_lstm_hidden_size, args.video_out, \
                            num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        # the classify layer for text
        self.post_video_dropout = nn.Dropout(p=args.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(args.video_out, args.swap_dim)

        self.feature = nn.Sequential(
            nn.Linear(args.swap_dim * 2, args.swap_dim),
            nn.ReLU(),
        )
        self.final = nn.Linear(args.swap_dim, 1)

    def forward(self, video, video_lengths):

        video = self.model_video(video, video_lengths)

        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        return video_h


class video_model(nn.Module):
    def __init__(self, args):
        super(video_model, self).__init__()
        self.args = args

        self.model_c = video_model_base(args)
        self.model_b = video_model_base(args)
        self.criterion = nn.L1Loss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(args, q=args.q)

    def forward(self, video, video_lengths, labels, epoch=0):
        batch_scores_c = self.model_c(video, video_lengths)
        batch_scores_b = self.model_b(video, video_lengths)
        z_c = torch.cat((batch_scores_c, batch_scores_b.detach()), dim=1)
        z_b = torch.cat((batch_scores_c.detach(), batch_scores_b), dim=1)
        batch_feature_c = self.model_c.feature(z_c)
        batch_feature_b = self.model_b.feature(z_b)
        # batch_scores_c1 = self.model_c.final(batch_feature_c)
        batch_scores_b1 = self.model_b.final(batch_feature_b)
        
        # loss_c = self.criterion(batch_scores_c1, labels).detach()
        loss_b = self.criterion(batch_scores_b1, labels).detach()
        
        # loss_weight = loss_b / (loss_b + loss_c + 1e-8)
        # loss_dis_conflict = self.criterion(batch_scores_c1, labels) * loss_weight.to(self.args.device)
        #print("******************", loss_dis_conflict, loss_weight)
        loss_dis_align = self.bias_criterion(batch_scores_b1, labels)
        if epoch > self.args.swap_epochs:
            indices = np.random.permutation(batch_scores_b.size(0))
            z_b_swap = batch_scores_b[indices]         # z tilde
            label_swap = labels[indices]     # y tilde
            # Prediction using z_swap=[z_l, z_b tilde]
            # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
            z_mix_conflict = torch.cat((batch_scores_c, z_b_swap.detach()), dim=1)
            z_mix_align = torch.cat((batch_scores_c.detach(), z_b_swap), dim=1)
            # Prediction using z_swap
            pred_mix_conflict = self.model_c.feature(z_mix_conflict)
            #pred_mix_conflict = self.model_c.final(pred_mix_conflict)
            pred_mix_align = self.model_b.feature(z_mix_align)
            pred_mix_align = self.model_b.final(pred_mix_align)
            # loss_swap_conflict = self.criterion(pred_mix_conflict, labels) * loss_weight.to(self.args.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
            loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
            lambda_swap = self.args.lambda_swap                                         # Eq.3 lambda_swap_b
        else:
            # before feature-level augmentation
            loss_swap_conflict = torch.tensor([0]).float()
            loss_swap_align = torch.tensor([0]).float()
            lambda_swap = 0
            pred_mix_conflict = 0

        # loss_swap = loss_swap_conflict.mean() + args.lambda_dis*loss_swap_align.mean()
        # loss_dis = loss_dis_conflict.mean() + args.lambda_dis*loss_dis_align.mean()
        # loss = loss_dis + lambda_swap * loss_swap #+ 1e-10*independ_loss #+ 0.00001*l1_loss
        loss = lambda_swap * self.args.lambda_dis*loss_swap_align.mean() + self.args.lambda_dis*loss_dis_align.mean()

        return loss, loss_b, batch_feature_c, pred_mix_conflict

class fusion_model_base(nn.Module):
    def __init__(self, args):
        super(fusion_model_base, self).__init__()
        self.args = args

        self.criterion = nn.L1Loss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(args, q=args.q)
        self.post_fusion_dropout = nn.Dropout(p=args.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(3 * args.swap_dim, args.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(args.post_fusion_dim, args.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(args.post_fusion_dim, 1)

    def forward(self, text, audio, video, labels, epoch=0):
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)
        loss = self.criterion(output_fusion, labels)

        return loss, output_fusion
    
class fusion_model_transformer(nn.Module):
    def __init__(self, args):
        super(fusion_model_transformer, self).__init__()
        self.args = args

        self.criterion = nn.L1Loss(reduction='none')
        self.bias_criterion = GeneralizedCELoss(args, q=args.q)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.swap_dim, nhead=args.trans_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.trans_layers)
        self.activation = nn.ReLU()

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=args.swap_dim * 3, out_features=args.post_fusion_dim))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(0.1))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=args.post_fusion_dim, out_features=1))

    def forward(self, text, audio, video, labels, text_s, audio_s, video_s, epoch=0):
        # fusion
        h = torch.stack((text, audio, video), dim=0)
        # print(h.shape) # 3,bs,32
        h = self.transformer_encoder(h)
        # print(h.shape) # 3,bs,32
        h = torch.cat((h[0], h[1], h[2]), dim=1)
        # print(h.shape) # bs,96
        output_fusion = self.fusion(h)

        loss = self.criterion(output_fusion, labels)

        if epoch > self.args.swap_epochs:
            h_s = torch.stack((text_s, audio_s, video_s), dim=0)
            h_s = self.transformer_encoder(h_s)
            h_s = torch.cat((h_s[0], h_s[1], h_s[2]), dim=1)
            output_fusion_s = self.fusion(h_s)
            loss_s = self.criterion(output_fusion_s, labels)
        else:
            loss_s = torch.tensor([0]).float().to(self.args.device)
        
        loss_ = loss + self.args.lambda_swap * loss_s

        return loss_, output_fusion

class IFL_ES(nn.Module):
    def __init__(self, args):
        super(IFL_ES, self).__init__()
        self.text_model = text_model(args)
        self.audio_model = audio_model(args)
        self.video_model = video_model(args)
        self.fusion_model = fusion_model_transformer(args)
        # self.fusion_model = fusion_model_base(args)
        # self.fusion_model = fusion_model_self_mm(args)

class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1
