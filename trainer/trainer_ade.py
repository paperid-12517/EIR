import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import json
import random
import heapq
from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from utils import transformsgpu
from utils import transformmasks
from models.loss import WBCELoss, KDLoss, ACLoss, MBCELoss, BCEWithLogitsLossWithIgnoreIndex
from data_loader import ADE


def strongTransform1(parameters, data=None, target=None, position=None, mask_n=None,width=None,height=None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mix_bg(mask=parameters["p_Mix_bg"], data=data, target=target, position=position, mask_n=mask_n) #mask: memory中对应某个伪标签的类别的掩码   mask_n: image_i中所有类别对应的掩码  data[0]: mamory中对应的图片 data[1]: 被贴的图片
    return data, target

def strongTransform2(parameters, data=None, target=None, position=None, mask_n=None,width=None,height=None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.Mix_up_bg(mask=parameters["Mix_up_bg"], data=data, target=target, num1=num1)
    return data, target

def strongTransform3(parameters, data=None, target=None, position=None, mask_nn=None,width=None,height=None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mix_beijing(mask=parameters["mix_beijing"], data=data, target=target, mask_nn=mask_nn,num1=num1)
    return data, target


def strongTransform4(parameters, data=None, target_0=None,target_1=None, mask_nn=None,found_region = None,num1=21):
    assert ((data is not None) or (target is not None))
    data, target = transformsgpu.p_Mixup_beijing(mask=parameters["mixup_beijing"], data=data, target_0=target_0, target_1=target_1, mask_nn=mask_nn, found_region = found_region, num1=num1)
    return data, target

class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 100-50: 100 | 100-10: 100 | 50-50: 50 |

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_ac',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        pos_weight = torch.ones([len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
        #self.BCELoss = WBCELoss(pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.BCELoss = BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, reduction='mean')
        self.ACLoss = ACLoss()

        self._print_train_info()

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['ac']} * L_ac")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit,  # [N, |C1:t|, H, W]
                    data['label'],                   # [N, H, W]
                )

                loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])  # [1]

                loss = self.config['hyperparameter']['mbce'] * loss_mbce + self.config['hyperparameter']['ac'] * loss_ac.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.item())
            self.train_metrics.update('loss_ac', loss_ac.mean().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _valid_epoch(self, epoch):
        torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
        return log

    def _test(self, epoch=None):
        torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit[:, 1:].argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit[:, 1:] > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': f"{met()['new']:.2f}"})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': f"{met()['harmonic']:.2f}"})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, memory_set, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)
        self.memory_set = memory_set
        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        memory_json = 'models' + '/' + f"overlap_{self.config['data_loader']['args']['task']['name']}_DKD" + '/' + f"step_{self.config['data_loader']['args']['task']['step']}" + '/'+"memory.json"

        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)
        file_names = memory_list[f"step_{self.config['data_loader']['args']['task']['step']}"]["memory_list"]
        file_xinxi = memory_list[f"step_{self.config['data_loader']['args']['task']['step']}"]["memory_xinxi"]  
    ## define a dict to record class information  begin
        self.memory_labelxinxi =  {i: [] for i in range(150+1)}   #voc : 20   ade:150
        for file_name in file_names:
            labels = file_xinxi[file_name][0]
            #print(labels)
            if 255 in labels:
                labels.remove(255)
            for label in labels: 
                #print(label)
                self.memory_labelxinxi[label].append(file_name)         


        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_kd', 'loss_dkd_pos', 'loss_dkd_neg', 'loss_ac',
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.KDLoss = KDLoss(pos_weight=None, reduction='none')

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + {self.config['hyperparameter']['kd']} * L_kd "
                         f"+ {self.config['hyperparameter']['dkd_pos']} * L_dkd_pos + {self.config['hyperparameter']['dkd_neg']} * L_dkd_neg "
                         f"+ {self.config['hyperparameter']['ac']} * L_ac")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad(set_to_none=True)
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
               
                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old = self.model_old(data['image'], ret_intermediate=True)
    
                pred_prob = torch.sigmoid(logit_old).detach()
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where(
                        (data['label'] <= 0) & (pred_labels > 0) & (pred_scores >= 0.7),
                        pred_labels,
                        data['label'])
                labels_copy = data['label'].clone().to(self.device, non_blocking=True)     
                data['label'] = pseudo_labels        

                if self.config['mix_mask'] == 'mixup_beijing':
                    num1 = self.n_old_classes+self.n_new_classes+1  
                    n_cl = torch.tensor(num1).cuda()
                    labels_new_1 = torch.where(data['label'] != 255, data['label'], n_cl)  
                    data['label'] = F.one_hot(labels_new_1, num1 + 1).float().permute(0, 3, 1, 2)
                    data['label'] = data['label'][:, :num1, :, :]          
                    for image_i in range(data['image'].size(0)):
                        unique_classes_n = torch.unique(labels_copy[image_i]).tolist()      
                        unique_classes_p = torch.unique(pseudo_labels[image_i]).tolist()   

                        set_n = set(unique_classes_n)
                        set_p = set(unique_classes_p)
                        common_classes = list(set_n.intersection(set_p))     #get same category: common_classes
                        different_classes_p = list(set_p.difference(set_n))  #get different category :different_classes_p;
                        
                        if len(different_classes_p) > 3:
                            different_classes_p = different_classes_p[:3]

                        
                        unique_classes_y = torch.unique(labels_copy[image_i])         
                        classes_yt = unique_classes_y[(unique_classes_y != 0) & (unique_classes_y != 255)]
                        mask_n = transformmasks.generate_class_mask(labels_copy[image_i],classes_yt).cuda()       
                        dengfen_num = 200

                        max_heap = []               
                        max_beijing = float('-inf')  
                        max_regions = 3              

                        found_region = [0,0]
                        #visited_regions = set()
                        for i in range(0, mask_n.shape[0] - dengfen_num - 1, dengfen_num):
                            for j in range(0, mask_n.shape[0] - dengfen_num - 1, dengfen_num):
                                region = mask_n[i:i+dengfen_num, j:j+dengfen_num]
                                num_0 = (region == 0).sum().item()
                                region_coord = (i,j)

                                #if region_coord not in visited_regions:
                                heapq.heappush(max_heap, (-num_0, region_coord))  
                                    #visited_regions.add(region_coord)  
              
                                if len(max_heap) > max_regions:  
                                    heapq.heappop(max_heap)  
           
                        top_regions = [(-val, coord) for val, coord in max_heap] 
                        i = 0
                        for class_p in different_classes_p:
                            #first method     in main_copy_24_4_12.py  
                            num_0,(found_region[0],found_region[1]) = top_regions[i]
                            i = i+1           
                            #second method                        
                            randnum0 = random.randint(0, len(self.memory_labelxinxi[class_p])-1)   
                            i_name = self.memory_labelxinxi[class_p][randnum0]   #image name
                            
                            sample_total = self.memory_set.get_image_by_name(i_name)    #get image by image_name
                            m_img,m_lab,name6 = sample_total['image'],sample_total['label'], sample_total['image_name']
                            #check m_lab class
                            check = torch.unique(m_lab)    #check： class_p=1  check:tensor([0,1])                     
                            class_pt = torch.tensor(class_p).unsqueeze(0).to(self.device, dtype=torch.long, non_blocking=True)  
                            m_img = m_img.to(self.device, dtype=torch.float32, non_blocking=True)
                            m_lab = m_lab.to(self.device, dtype=torch.long, non_blocking=True)
                            MixMask = transformmasks.generate_class_mask(m_lab, class_pt).unsqueeze(0).cuda()   #[1,513,513]
                            
                            strong_parameters = {"mixup_beijing": MixMask}     

                            data['image'][image_i], data['label'][image_i] = strongTransform4(strong_parameters,data=torch.cat((m_img.unsqueeze(0), data['image'][image_i].unsqueeze(0))), target_0 = m_lab, target_1 = data['label'][image_i], mask_nn = None,found_region=found_region, num1=num1)

                        if len(different_classes_p) < 1:
                            #num = 0
                            num_0,(found_region[0],found_region[1]) = top_regions[0]
                            inte = 3 - len(different_classes_p)
                            for i in range(0,inte):
                                randnum_class = random.randint(1, self.n_old_classes)  
                                randnum0 = random.randint(0, len(self.memory_labelxinxi[randnum_class])-1)  
                                i_name = self.memory_labelxinxi[randnum_class][randnum0]  
                                sample_total = self.memory_set.get_image_by_name(i_name)    #get image by image_name
                                m_img,m_lab,name6 = sample_total['image'],sample_total['label'], sample_total['image_name']
                                #check m_lab class
                                #check = torch.unique(m_lab)    #check： randnum_class=1  check:tensor([0,1])          randnum_class
                                randnum_classt = torch.tensor(randnum_class).unsqueeze(0).to(self.device, dtype=torch.long, non_blocking=True)  
                                m_img = m_img.to(self.device, dtype=torch.float32, non_blocking=True)
                                m_lab = m_lab.to(self.device, dtype=torch.long, non_blocking=True)
                                MixMask = transformmasks.generate_class_mask(m_lab, randnum_classt).unsqueeze(0).cuda() 
                        

                                strong_parameters = {"mixup_beijing": MixMask}       
                                data['image'][image_i], data['label'][image_i] = strongTransform4(strong_parameters,data=torch.cat((m_img.unsqueeze(0), data['image'][image_i].unsqueeze(0))), target_0 = m_lab, target_1 = data['label'][image_i], mask_nn = None,found_region=found_region,num1=num1)

                logit, features = self.model(data['image'], ret_intermediate=True)
                if self.model_old is not None:
                    with torch.no_grad():
                        logit_old, features_old = self.model_old(data['image'], ret_intermediate=True)

                num1 = self.n_old_classes+self.n_new_classes+1
                n_cl = torch.tensor(num1).cuda()
                labels_new_1 = torch.where(data['label'] != 255, data['label'], n_cl) 
                if labels_new_1.dtype != torch.long:  
                    labels_new_1 = labels_new_1.long()
                data['label'] = F.one_hot(labels_new_1, num1 + 1).float().permute(0, 3, 1, 2)
                data['label'] = data['label'][:, :num1, :, :]  # remove 255 from 1hot   (2,12,513,513)
                

                loss_mbce = self.BCELoss(
                    logit,  # [N, |C1:t|, H, W]
                    data['label'],                   # [N, H, W]
                )

                # [|C0:t-1|]
                loss_kd = self.KDLoss(
                    logit[:, 1:self.n_old_classes + 1],  # [N, |C0:t|, H, W]
                    logit_old[:, 1:].sigmoid()       # [N, |C0:t|, H, W]
                ).mean(dim=[0, 2, 3])

                # [1]
                loss_ac = self.ACLoss(logit[:, 0:1]).mean(dim=[0, 2, 3])

                # [|C0:t-1|]
                loss_dkd_pos = self.KDLoss(
                    features['pos_reg'][:, :self.n_old_classes],
                    features_old['pos_reg'].sigmoid()
                ).mean(dim=[0, 2, 3])

                # [|C0:t-1|]
                loss_dkd_neg = self.KDLoss(
                    features['neg_reg'][:, :self.n_old_classes],
                    features_old['neg_reg'].sigmoid()
                ).mean(dim=[0, 2, 3])

                loss = self.config['hyperparameter']['mbce'] * loss_mbce + self.config['hyperparameter']['kd'] * loss_kd.sum() + \
                    self.config['hyperparameter']['dkd_pos'] * loss_dkd_pos.sum() + self.config['hyperparameter']['dkd_neg'] * loss_dkd_neg.sum() + \
                    self.config['hyperparameter']['ac'] * loss_ac.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.item())
            self.train_metrics.update('loss_kd', loss_kd.mean().item())
            self.train_metrics.update('loss_ac', loss_ac.mean().item())
            self.train_metrics.update('loss_dkd_pos', loss_dkd_pos.mean().item())
            self.train_metrics.update('loss_dkd_neg', loss_dkd_neg.mean().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag
