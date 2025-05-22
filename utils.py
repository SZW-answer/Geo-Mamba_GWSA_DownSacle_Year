import os
from tqdm import tqdm
import torch
from torch.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import itertools
from torch.utils.tensorboard import SummaryWriter
# 初始化 TensorBoard
writer = SummaryWriter(log_dir='runs/attention_and_feature_visualization')
log_interval = 100  # 
import matplotlib.pyplot as plt
# 字典用于存储激活

def evaluate_model(model, test_loader, criterion, accelerator,use_amp=True):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for dynamic_x, lat, lon, static_features, year,month,targets in tqdm(test_loader, desc="Evaluating"):
            if use_amp:
               with accelerator.autocast():
                    outputs = model(dynamic_x, lat, lon, static_features,year,month)
                    loss = criterion(accelerator.gather(outputs), accelerator.gather(targets))
            else:
                outputs = model(dynamic_x, lat, lon, static_features,year,month)
                loss = criterion(accelerator.gather(outputs), accelerator.gather(targets))
            total_loss += loss.item() * dynamic_x.size(0)
    avg_loss = total_loss / len(test_loader.dataset)
    accelerator.print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model_metrics(model, test_loader,  accelerator,use_amp=True):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for dynamic_x, lat, lon, static_features, year,month,targets in tqdm(test_loader, desc="Evaluating Metrics"):
            if use_amp:
                with accelerator.autocast():
                    outputs = model(dynamic_x, lat, lon, static_features,year,month)
            else:
                outputs = model(dynamic_x, lat, lon, static_features,year,month)
            outputs = accelerator.gather(outputs)
            targets =accelerator.gather(targets)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    accelerator.print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return mse, rmse, mae, r2


def train_model_with_evaluation(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        accelerator,
        max_iterations=1000,  # 替换 epochs
        eval_every_iters=100,  # 替换 eval_every
        save_every_iters=100,  # 替换 eval_every2
        log_file='training_log.txt',
        checkpoint_path='latest_checkpoint.pth',
        save_dir='checkpoints',
        use_amp=True,  # 继续使用AMP的选项
        
):
    # 获取 Accelerator 对象
    accelerator =accelerator
    from accelerate.utils import set_seed
    set_seed(42)
        # # 准备模型、优化器、数据加载器和调度器
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )
    
    
    best_r2 = np.inf  # 初始化最佳R²

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader_iter = itertools.cycle(train_loader)  # 创建一个无限循环的迭代器
    current_iter = 0

    # 如果需要从checkpoint恢复，这部分代码需要根据`max_iterations`进行调整
    # 在这里假设您从头开始训练

    while current_iter < max_iterations:
        current_iter += 1
        dynamic_x, lat, lon, static_features, year,month,targets = next(train_loader_iter)
        optimizer.zero_grad()
        from torchinfo import summary
        # 打印模型结构
        print("Model Architecture:")
        summary(
            model,
            input_data=(dynamic_x, lat, lon, static_features, year, month),
            device=accelerator.device,
            verbose=1
        )
        
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        # 在相同位置添加：
        # 计算 FLOPs
        flops = FlopCountAnalysis(model, (dynamic_x, lat, lon, static_features, year, month))
        print("FLOP Details:")
        print(flop_count_table(flops))

        # 转换为 GFLOPs 并按层输出
        print("\nGFLOPs per Layer:")
        for name, flop in flops.by_module().items():
            print(f"{name:30s} | {flop / 1e9:.4f} GFLOPs")
        
        
        
        if use_amp:
            with accelerator.autocast():
                outputs = model(dynamic_x, lat, lon, static_features,year,month)
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
        else:
            outputs = model(dynamic_x, lat, lon, static_features,year,month)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)

        optimizer.step()
        
        loss_value = loss.detach()  # 避免梯度干扰
        epoch_loss = accelerator.reduce(loss_value, reduction="mean").item()  
        # epoch_loss = loss.item()
        scheduler.step()

        # 打印训练信息
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        if current_iter % 10 == 0 or current_iter == 1:
            accelerator.print(f"Iteration {current_iter}/{max_iterations}, Train Loss: {epoch_loss:.4f}, Learning Rate: {current_lr:.10f}")

        # 保存模型
        if current_iter % save_every_iters == 0:
            model_save_path = os.path.join(save_dir, f'model_iter_{current_iter}.pth')
            accelerator.save(accelerator.get_state_dict(model), model_save_path)
            accelerator.print(f"已保存模型: {model_save_path}")

        # 评估模型
        if current_iter % eval_every_iters == 0:
            accelerator.print(f"=== 第 {current_iter} 次迭代的评估开始 ===")
            test_loss = evaluate_model(model, test_loader, criterion, accelerator, use_amp=use_amp)
            mse, rmse, mae, r2 = evaluate_model_metrics(model, test_loader,  accelerator,use_amp=use_amp)

            # 判断是否为最佳模型
            is_best = False
            if rmse < best_r2:
                best_r2 = rmse
                is_best = True
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                accelerator.save(accelerator.get_state_dict(model), best_model_path)
                accelerator.print(f"新最佳Rmse: {best_r2:.4f}，已保存最佳模型: {best_model_path}")
                # 保存当前模型的checkpoint
                checkpoint = {
                    'iter': current_iter,
                    'model_state_dict':  accelerator.get_state_dict(model),
                    'optimizer_state_dict': accelerator.get_state_dict(optimizer),
                    'scheduler_state_dict': accelerator.get_state_dict(scheduler) if scheduler else None,
                    'best_r2': best_r2,
                    'scaler_state_dict': accelerator.get_state_dict(accelerator.scaler) if use_amp else None
                }
                checkpoint_best_model = os.path.join(save_dir, f"best_model_iter_{current_iter}_rmse_{best_r2:.4f}.pth")
                accelerator.save(checkpoint, checkpoint_best_model)
                accelerator.print(f"已保存最佳的RMSe的checkpoint: {checkpoint_best_model}")

            # 保存当前模型的checkpoint
            checkpoint = {
                'iter': current_iter,
                'model_state_dict': accelerator.get_state_dict(model),
                'optimizer_state_dict': accelerator.get_state_dict(optimizer),
                'scheduler_state_dict': accelerator.get_state_dict(scheduler) if scheduler else None,
                'best_r2': best_r2,
                'scaler_state_dict': accelerator.get_state_dict(accelerator.scaler) if use_amp else None
            }
            accelerator.save(checkpoint, checkpoint_path)
            accelerator.print(f"已保存最新的checkpoint: {checkpoint_path}")

            # 记录评估结果到日志文件
            with open(log_file, 'a') as f:
                f.write(
                    f"{current_iter}\t{epoch_loss:.4f}\t{test_loss:.4f}\t{mse:.4f}\t{rmse:.4f}\t{mae:.4f}\t{r2:.4f}\t{best_r2:.4f}\n")
            accelerator.print(f"=== 第 {current_iter} 次迭代的评估完成 ===\n")

    accelerator.print("训练完成。")

    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path,accelerator, use_amp=True):
    scaler = accelerator.scaler if use_amp else None  # 根据use_amp决定是否初始化GradScaler
    if os.path.exists(checkpoint_path):
        accelerator.print(f"找到checkpoint文件: {checkpoint_path}，开始加载...")
        checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if use_amp and scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])  # 加载 scaler 状态
        start_iter = checkpoint['iter'] + 1
        best_r2 = checkpoint.get('best_r2', np.inf)
        accelerator.print(f"已加载checkpoint，继续从Iteration {start_iter} 训练，最佳R²: {best_r2:.4f}")
        return model, optimizer, scheduler, start_iter, best_r2
    else:
        accelerator.print("未找到任何checkpoint文件，开始全新训练。")
        return model, optimizer, scheduler, 1, np.inf
