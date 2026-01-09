import copy
import logging
import random
import time
import torch
import train
import test
from gan import gan_defense
from models.ResNet8 import ResNet8
from device import device
logger = logging.getLogger("logger")


def get_clients(epoch, handle):
    adversarial_name_keys = []
    if handle.params['is_poison'] is True:
        agent_name_keys = random.sample(handle.participants_list, handle.params['num_clients'])
        for client in agent_name_keys:
            if client in handle.adversarial_namelist:
                adversarial_name_keys.append(client)
    else:
        agent_name_keys = random.sample(handle.benign_namelist, handle.params['num_clients'])
    logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
    return agent_name_keys, adversarial_name_keys


def Aggregation(w_ori, w_list, lens, beta, eta, defence_method, params):
    """优化聚合函数，使用向量化操作减少循环"""
    # 预计算所有键，避免在循环中重复获取
    keys = [k for k in w_ori.keys() if w_ori[k].dtype != torch.int64]
    
    # 使用向量化操作替代循环
    with torch.no_grad():
        # 计算加权差异
        weighted_diffs = []
        for i in range(len(w_list)):
            diff_dict = {}
            for k in keys:
                diff_dict[k] = (w_list[i][k] - w_ori[k]) * beta[i]
            weighted_diffs.append(diff_dict)
        
        # 计算平均值
        w_avg = {}
        for k in keys:
            stacked = torch.stack([wd[k] for wd in weighted_diffs])
            w_avg[k] = torch.mean(stacked, dim=0)
        
        # 更新原始权重
        for k in keys:
            w_ori[k] += eta * w_avg[k]
    
    return w_ori


def FedAvg(handle):
    count_all_attack = 0
    count_correct_defense = 0
    count_all_defense = 0
    
    # 预加载模型到设备，避免重复操作
    handle.model.to(device)
    
    for epoch in range(handle.start_epoch, handle.params['epochs'] + 1):
        if handle.params['lr_decay'] is True:
            if epoch % handle.params['lr_decay_epoch'] == 0:
                handle.params['lr'] = handle.params['lr'] * handle.params['lr_decay_gamma']
                handle.params['poison_lr'] = handle.params['poison_lr'] * handle.params['lr_decay_gamma']
        
        start_time = time.time()
        agent_name_keys, adversarial_name_keys = get_clients(epoch, handle)
        lens = len(agent_name_keys)
        count_all_attack += len(adversarial_name_keys)

        beta = [1] * lens
        ori_weight = handle.model.state_dict()
        w_locals = []
        
        # 使用列表推导式预分配空间
        for client in agent_name_keys:
            is_poison = client in adversarial_name_keys
            # 使用模型副本而不是深度拷贝
            model_copy = copy.deepcopy(handle.model)
            w = train.standard_train(epoch, handle.clients_data_num[client], client, handle.params,
                                     model_copy.to(device),
                                     handle.train_data[client], handle, poison=is_poison)
            w_locals.append(w)
        
        # 减少深度拷贝
        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()}, 
                            w_locals, lens, beta, handle.params['eta'],
                            handle.params['defence_method'], handle.params)

        if handle.params['defence_method'] == 'ours':
            standard = handle.params['ours_standard'][(epoch - 1) // 500]
            g_imgs = gan_defense(ori_weight, w_glob, handle.params, the_epoch=epoch)
            
            # 预加载模型到设备
            ResNet8_model = ResNet8(num_classes=10).to(device)
            
            for i in range(handle.params['num_clients']):
                if beta[i] == 0:
                    continue
                
                # 减少深度拷贝
                parameter = {}
                for k in w_locals[i].keys():
                    if k in ori_weight and ori_weight[k].dtype != torch.int64:
                        parameter[k] = (w_locals[i][k] - ori_weight[k]) * beta[i] + ori_weight[k]
                    else:
                        parameter[k] = w_locals[i][k]
                
                ResNet8_model.load_state_dict(parameter)
                
                with torch.no_grad():
                    out = torch.softmax(ResNet8_model(g_imgs[0]), dim=1)
                    out = torch.mean(out, dim=0)
                    out_val, out_n = torch.max(out, dim=0)
                    logger.info('{}, {}, {}'.format(agent_name_keys[i], out_n, out_val))
                    
                    if standard < out_val and out_n == 0:
                        beta[i] = 0
            
            # 更新统计信息
            for i in range(len(agent_name_keys)):
                if beta[i] == 0:
                    lens -= 1
                    count_all_defense += 1
                    logger.info(agent_name_keys[i])
                    if agent_name_keys[i] in adversarial_name_keys:
                        count_correct_defense += 1
            
            logger.info('count_all_attack: {}, count_all_defense: {}, count_correct_defense: {}'.format(
                count_all_attack, count_all_defense, count_correct_defense))
            
            if sum(beta) != 0:
                w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()}, 
                                    w_locals, lens, beta, handle.params['eta'],
                                    handle.params['defence_method'], handle.params)
            else:
                w_glob = ori_weight
        
        handle.model.load_state_dict(w_glob)
        acc = test.normal_test(handle.model, handle.test_data, handle.params, handle, poison=False)
        asr = test.normal_test(handle.model, handle.test_un_target_label_data, handle.params, handle, poison=True)
        
        # 记录 epoch 时间
        logger.info('Epoch {} completed in {:.2f} seconds'.format(epoch, time.time() - start_time))