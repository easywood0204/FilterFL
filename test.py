import logging
import torch
import torch.nn.functional as F
import main
from device import device
logger = logging.getLogger("logger")


def add_pixel_pattern(ori_image, params, handle):
    """优化像素模式添加函数，减少拷贝操作"""
    image = ori_image.clone()  # 使用clone替代deepcopy
    
    # 预计算所有毒模式
    poison_patterns = []
    for index in range(params['trigger_num']):
        pattern_key = f'{index}_poison_pattern'
        if pattern_key in params:
            poison_patterns.extend(params[pattern_key])
    
    if not poison_patterns:
        return image, handle.target
    
    # 批量处理像素模式
    if params['type'] != 'MNIST':
        for pos in poison_patterns:
            # 一次性设置所有通道
            image[:, pos[0], pos[1]] = 1
    else:
        for pos in poison_patterns:
            image[0, pos[0], pos[1]] = 1
    
    return image, handle.target


def normal_test(model, data_loader, params, handle, poison=False):
    """优化测试函数，减少不必要的操作"""
    model.eval()
    test_loss = 0.0
    correct = 0
    data_num = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            batch_size = len(target)
            
            if poison:
                # 批量处理投毒
                for i in range(batch_size):
                    data[i], target[i] = add_pixel_pattern(data[i], params, handle)
            
            # 使用非阻塞传输
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            log_probs = model(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            
            # 使用更高效的argmax
            pred = log_probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            data_num += batch_size
    
    test_loss /= data_num
    accuracy = 100.00 * correct / data_num
    
    logger.info('___Global_Test_Poison:{}  Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)'.format(
        poison, test_loss, correct, data_num, accuracy))
    
    return accuracy