import logging
import os
import pickle as plk
from pickletools import optimize

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ...utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class IFL_LS():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = "MTAV"
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def get_optimizer_text(self, model):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.model_c.model_text.named_parameters()) + list(model.model_b.model_text.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'model_text' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        return optimizer

    def get_optimizer_audio(self, model):
        audio_params = list(model.model_c.model_audio.named_parameters()) + list(model.model_b.model_audio.named_parameters())
        audio_params = [p for n, p in audio_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'model_audio' not in n]

        optimizer_grouped_parameters = [
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        return optimizer

    def get_optimizer_video(self, model):
        video_params = list(model.model_c.model_video.named_parameters()) + list(model.model_b.model_video.named_parameters())
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'model_video' not in n]

        optimizer_grouped_parameters = [
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        return optimizer
    
    def get_optimizer(self, model):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return optimizer

    def do_train(self, model, dataloader, return_epoch_results=False):
        model_text = model.Model.text_model
        model_audio = model.Model.audio_model
        model_video = model.Model.video_model
        model_fusion = model.Model.fusion_model

        optimizer_text = self.get_optimizer_text(model_text)
        optimizer_audio = self.get_optimizer_audio(model_audio)
        optimizer_video = self.get_optimizer_video(model_video)
        optimizer_fusion = self.get_optimizer(model_fusion)

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        outputs = {}
        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            ids = []
            epoch_results = {}
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer_text.zero_grad()
                        optimizer_audio.zero_grad()
                        optimizer_video.zero_grad()
                        optimizer_fusion.zero_grad()

                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    loss_text_gce, loss_text_b, outputs['T'], outputs['Feature_t'] = model_text(text, labels)
                    loss_audio_gce, loss_audio_b, outputs['A'], outputs['Feature_a'] = model_audio(audio, audio_lengths,\
                                                                                                     labels)
                    loss_video_gce, loss_video_b, outputs['V'], outputs['Feature_v'] = model_video(vision, vision_lengths,\
                                                                                                    labels)
                    loss_fusion, outputs['M'] = model_fusion(outputs['Feature_t'], outputs['Feature_a'], \
                                                             outputs['Feature_v'], labels, loss_text_b, \
                                                             loss_audio_b, loss_video_b, epochs)
                    
                    loss = loss_text_gce + loss_audio_gce + loss_video_gce + loss_fusion
                    
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels.cpu())
                        
                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer_text.step()
                        optimizer_audio.step()
                        optimizer_video.step()
                        optimizer_fusion.step()

                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer_text.step()
                    optimizer_audio.step()
                    optimizer_video.step()
                    optimizer_fusion.step()
            train_loss = train_loss / len(dataloader['train'])
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >> loss: {round(train_loss, 4)}"
            )
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        model_text = model.Model.text_model
        model_audio = model.Model.audio_model
        model_video = model.Model.video_model
        model_fusion = model.Model.fusion_model

        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        outputs = {}
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths']
                        vision_lengths = batch_data['vision_lengths']
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    # outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss_text_gce, loss_text_b, outputs['T'], outputs['Feature_t'] = model_text(text, labels)
                    loss_audio_gce, loss_audio_b, outputs['A'], outputs['Feature_a'] = model_audio(audio, audio_lengths,\
                                                                                                     labels)
                    loss_video_gce, loss_video_b, outputs['V'], outputs['Feature_v'] = model_video(vision, vision_lengths,\
                                                                                                    labels)
                    loss_fusion, outputs['M'] = model_fusion(outputs['Feature_t'], outputs['Feature_a'], \
                                                             outputs['Feature_v'], labels, \
                                                             loss_text_b=torch.ones_like(loss_text_b), \
                                                             loss_audio_b=torch.ones_like(loss_audio_b), \
                                                             loss_video_b=torch.ones_like(loss_video_b))
                    
                    loss = loss_text_gce + loss_audio_gce + loss_video_gce + loss_fusion

                    # if return_sample_results:
                    #     ids.extend(batch_data['id'])
                    #     for item in features.keys():
                    #         features[item].append(outputs[item].cpu().detach().numpy())
                    #     all_labels.extend(labels_m.cpu().detach().tolist())
                    #     preds = outputs["M"].cpu().detach().numpy()
                    #     # test_preds_i = np.argmax(preds, axis=1)
                    #     sample_results.extend(preds.squeeze())

                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.model_name + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = round(eval_loss, 4)

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results