import logging
import torch
from torch import nn
from device import device  # 你的 device 定义（例如 torch.device("cuda") / cpu）
logger = logging.getLogger("logger")

# 预计算毒化模式mask，避免重复计算
# 注意：现在缓存的 mask 会和生成它的 device 绑定（key 包含 device），
# 以避免频繁在 CPU/GPU 之间拷贝。
_pattern_masks = {}

def get_pattern_mask(params, index, image_shape, dev):
    """
    预计算毒化模式 mask 并缓存到指定设备 dev 上。
    image_shape: e.g. (C,H,W)
    dev: torch.device 或 device string
    """
    # 统一 key，包含设备信息
    key = f"{index}_{params.get('type','')}_{image_shape}_{str(dev)}"
    if key in _pattern_masks:
        return _pattern_masks[key]

    # 先在目标 device 上创建 zeros mask
    mask = torch.zeros(image_shape, device=dev, dtype=torch.uint8)

    poison_patterns = params.get(f'{index}_poison_pattern', [])
    if not poison_patterns:
        _pattern_masks[key] = mask
        return mask

    if params.get('type', '') != 'MNIST':
        for pos in poison_patterns:
            # pos expected (row, col) or similar
            mask[:, pos[0], pos[1]] = 1
    else:
        for pos in poison_patterns:
            mask[0, pos[0], pos[1]] = 1

    _pattern_masks[key] = mask
    return mask


def apply_pattern_batch_gpu(images, params, index, handle, poison_mask):
    """
    在 GPU 上对整个 batch(images) 应用毒化 pattern。
    images: tensor (N,C,H,W) 已经在目标 device 上
    poison_mask: bool tensor (N,) 在同一 device 上，表示哪些 sample 要被毒化
    返回: (images_after, labels_tensor_for_poisoned)  - 都在相同 device 上
    """
    if images.size(0) == 0:
        return images, torch.full((0,), handle.target, device=images.device, dtype=torch.long)

    dev = images.device
    # 获取单样本 pattern mask (C,H,W) 已经在 dev 上
    pattern_mask = get_pattern_mask(params, index, images[0].shape, dev)  # uint8
    # 扩展到 batch
    # pattern_mask: (C,H,W) -> (1,C,H,W) -> expand to (N,C,H,W)
    batch_pattern_mask = pattern_mask.unsqueeze(0).expand(images.size(0), -1, -1, -1).bool()

    # poison_mask: (N,) -> (N,1,1,1) broadcastable
    poison_mask_exp = poison_mask.view(-1, 1, 1, 1)

    # 构造 replacement tensor (1.0) 与 images 形状一致
    # 使用 ones_like(images) 会消耗较多显存，所以下面只在 mask 为 True 的位置替换成 1.0
    # 使用 torch.where：条件是 (poison_mask_exp & batch_pattern_mask)
    replace_val = torch.tensor(1.0, device=dev, dtype=images.dtype)

    # 执行一次性替换（全部在 GPU 上）
    images_after = torch.where(poison_mask_exp & batch_pattern_mask, replace_val, images)

    # labels：对于被污染的样本设置为 handle.target，否则保留原 label（调用处需要合并）
    poisoned_labels = torch.full((images.size(0),), handle.target, device=dev, dtype=torch.long)

    return images_after, poisoned_labels


def _get_data_iterator(dataset):
    """
    兼容原来 dataset 结构：如果 dataset 是 (something, iterator)，返回 iterator，
    否则假设 dataset 本身是 iterable (DataLoader) 并返回它。
    """
    try:
        # 如果 dataset 是 tuple(pair)，且第二项是 data iterator
        if isinstance(dataset, (list, tuple)) and len(dataset) >= 2:
            return dataset[1]
    except Exception:
        pass
    return dataset


def standard_train(epoch, dataset_size, client, params, model, dataset, handle, poison=False):
    """
    优化后的训练函数（GPU 优先、AMP 支持、GPU 上批量毒化、延迟同步统计）
    参数与原来基本保持一致。
    """
    model.train()
    index = -1
    local_epoch = int(params.get('local_epochs', 1))
    lr = float(params.get('lr', 0.01))

    if poison:
        local_epoch = int(params.get('poison_epochs', local_epoch))
        lr = float(params.get('poison_lr', lr))
        index = 0

    # 将模型放到目标设备
    model.to(device)

    # 损失与优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(params.get('momentum', 0.9)),
        weight_decay=float(params.get('decay', 0.0))
    )

    data_iterator = _get_data_iterator(dataset)

    # 若 dataset 是 DataLoader，确保用户设置 pin_memory=True 和合理 num_workers，否则这一部分可能仍然是瓶颈。
    # poison_rate
    poison_rate = float(params.get('poison_rate', 0.0)) if poison else 0.0

    use_amp = (device.type == 'cuda') and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 训练循环
    for local_ep in range(local_epoch):
        # 在 GPU 上延迟统计（tensor）
        total_loss_tensor = torch.tensor(0.0, device=device)
        correct_tensor = torch.tensor(0, device=device, dtype=torch.long)
        processed = 0

        # iterate
        for batch_idx, batch in enumerate(data_iterator):
            # 兼容 dataloader 返回 (images, labels) 或 batch 本身就是 (images, labels)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                # 无法解析 batch，跳过
                continue

            # 防护：若 batch 空则跳过
            if images is None or labels is None:
                continue

            # 移动到设备（non_blocking 以配合 pin_memory）
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = images.size(0)
            if batch_size == 0:
                continue

            # --- 在 GPU 上执行毒化（一次性） ---
            if poison and poison_rate > 0.0:
                poison_num = int(batch_size * poison_rate)
                if poison_num > 0:
                    # 在 GPU 上生成布尔 mask
                    # 使用 randperm 在 GPU 上选取要毒化的样本
                    perm = torch.randperm(batch_size, device=device)
                    poison_indices = perm[:poison_num]
                    # 构造 poison_mask bool (N,)
                    poison_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
                    poison_mask[poison_indices] = True

                    # apply_pattern_batch_gpu 返回 (images_after, poisoned_labels)
                    images_after, poisoned_labels = apply_pattern_batch_gpu(images, params, index, handle, poison_mask)

                    # 仅替换被毒化样本的 labels（labels 仍在 GPU）
                    # labels = torch.where(poison_mask, poisoned_labels, labels)
                    # 为避免额外复制，直接如下（会在 GPU 上完成）
                    labels = torch.where(poison_mask, poisoned_labels, labels)

                    # 更新 images 引用
                    images = images_after

            # 前向/反向/优化 使用 AMP（若可用）
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_func(outputs, labels)
                # backward & step via scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

            # 延迟统计：在 GPU 上累加（避免 .item() 导致同步）
            total_loss_tensor += loss.detach() * batch_size
            preds = outputs.argmax(dim=1)
            correct_tensor += preds.eq(labels).sum()
            processed += batch_size

        # epoch 结束，移动统计到 cpu 并记录
        if processed == 0:
            avg_loss = float('nan')
            accuracy = float('nan')
        else:
            # 将 tensor 移到 cpu 并转换为 python float/int
            avg_loss = (total_loss_tensor / processed).cpu().item()
            accuracy = (100.0 * correct_tensor.float() / processed).cpu().item()

        logger.info('___Local_Train , g_epoch {:3d}, l_epoch {:3d}, local model {},  Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
            epoch, local_ep + 1, client, avg_loss, int(correct_tensor.cpu().item()), processed, accuracy))

    # 返回模型参数副本（与原来接口一致）
    return model.state_dict()
