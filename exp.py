import torch
import torch.nn as nn
from torch import optim
import os
import sys
import time
import warnings
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from iTransformer import iTransformer
from PaiFilter import PaiFilter
from Leddam import Leddam_model
from utils import setup_for_distributed, is_main_process, get_rank, synchronize, filter_channels
warnings.filterwarnings('ignore')

def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def metrics(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def get_data(args,flag='train'):
    if args.data in ['ETTh1', 'ETTh2']:
        data_set = Dataset_ETT_hour(args, root_path=args.root_path, flag=flag, data_path=args.data_path, features=args.features, size=[args.seq_len, args.label_len, args.pred_len])
    elif args.data in ['ETTm1', 'ETTm2']:
        data_set = Dataset_ETT_minute(args, root_path=args.root_path, flag=flag, data_path=args.data_path, features = args.features, size=[args.seq_len, args.label_len, args.pred_len])
    else:
        data_set = Dataset_Custom(args, root_path=args.root_path, flag=flag, data_path = args.data_path,size=[args.seq_len, args.label_len, args.pred_len],dates=args.dates)
    shuffle_flag = True if flag == 'train' else False
    
    if args.distributed:
        sampler = DistributedSampler(data_set, shuffle=True)
        data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=sampler, pin_memory=True, num_workers=args.num_workers)
    else:
        data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=shuffle_flag, pin_memory=True, num_workers=args.num_workers)
        
    return data_set, data_loader


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if get_rank() == 0: # Only rank 0 saves checkpoint
                self.save_checkpoint(val_loss, model, path)
                test(args,model,nn.MSELoss(), args.device) # Run test on initial best model
        elif score < self.best_score + self.delta:
            self.counter += 1
            if get_rank() == 0:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if get_rank() == 0: # Only rank 0 saves checkpoint
                self.save_checkpoint(val_loss, model, path)
                test(args,model,nn.MSELoss(), args.device)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose and get_rank() == 0:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Ensure path exists (useful if called by rank 0 only)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        model_to_save = model.module if isinstance(model, DDP) else model
        torch.save(model_to_save.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def iter_spectral_filters(model):
    model_to_scan = unwrap_model(model)
    for module in model_to_scan.modules():
        if hasattr(module, 'set_bypass') and hasattr(module, 'reset_parameters') and hasattr(module, 'complex_weight'):
            yield module


def set_spectral_filter_bypass(model, bypass):
    for spectral_filter in iter_spectral_filters(model):
        spectral_filter.set_bypass(bypass)


def reset_spectral_filters(model):
    for spectral_filter in iter_spectral_filters(model):
        spectral_filter.reset_parameters()


def set_filter_warmup_phase(model):
    set_spectral_filter_bypass(model, True)
    for spectral_filter in iter_spectral_filters(model):
        for param in spectral_filter.parameters():
            param.requires_grad = False


def set_filter_only_phase(model):
    set_spectral_filter_bypass(model, False)
    for param in unwrap_model(model).parameters():
        param.requires_grad = False
    reset_spectral_filters(model)
    for spectral_filter in iter_spectral_filters(model):
        for param in spectral_filter.parameters():
            param.requires_grad = True


def set_train_mode_for_phase(model, staged_filter_training, filter_phase_active):
    if staged_filter_training and filter_phase_active:
        model.eval()
        for spectral_filter in iter_spectral_filters(model):
            spectral_filter.train()
    else:
        model.train()


def build_optimizer_and_scheduler(args, model):
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found for the current training phase.")
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    return optimizer, scheduler


def load_best_checkpoint(args, model, device):
    best_model_pth = os.path.join(args.path, 'checkpoint.pth')
    if os.path.exists(best_model_pth):
        unwrap_model(model).load_state_dict(torch.load(best_model_pth, map_location=device))


def vali(args, vali_data, model, vali_loader, criterion, device):
    total_loss= []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            if args.exclude_experiment and args.exclude_channels:
                batch_x, batch_y = filter_channels(batch_x, batch_y, args.exclude_channels)

            if args.num_submodels > 1:
                start_idx_in = args.current_submodel_idx * args.sub_enc_in
                end_idx_in = (args.current_submodel_idx + 1) * args.sub_enc_in
                batch_x = batch_x[..., start_idx_in:end_idx_in]

                start_idx_out = args.current_submodel_idx * args.sub_c_out
                end_idx_out = (args.current_submodel_idx + 1) * args.sub_c_out
                batch_y = batch_y[..., start_idx_out:end_idx_out]
    
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs = model(batch_x)
            outputs = outputs[:,-args.pred_len:,:]
            batch_y = batch_y[:,-args.pred_len:,:]
            pred = outputs.detach() 
            true = batch_y.detach() 

            loss = criterion(pred,true)
            total_loss.append(loss.item()) 

    avg_loss = np.mean(total_loss)
    if args.distributed:
        loss_tensor = torch.tensor(avg_loss).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / args.world_size
        
    return avg_loss

def train(args, model, criterion,optimizer, device,scheduler):
    _,train_loader = get_data(args,flag='train')

    if args.distributed and is_main_process(): # Set sampler epoch for reproducibility if needed
        train_loader.sampler.set_epoch(0) # Example, set epoch at start of each epoch loop

    val_data,val_loader = get_data(args,flag='val')
    
    filter_warmup_epochs = getattr(args, 'filter_warmup_epochs', 0)
    staged_filter_training = filter_warmup_epochs > 0
    filter_phase_active = not staged_filter_training

    if is_main_process():
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        warmup_early_stopping = EarlyStopping(patience=args.patience, verbose=True) if staged_filter_training else None

    if staged_filter_training:
        set_filter_warmup_phase(model)
        optimizer, scheduler = build_optimizer_and_scheduler(args, model)
        if is_main_process():
            print(f"Staged filter training enabled: filters bypassed for up to {filter_warmup_epochs} epochs.")

    start = time.time()
    
    for epoch in range(args.epochs):
        if args.distributed: # Important for DistributedSampler to reshuffle data
            train_loader.sampler.set_epoch(epoch)

        set_train_mode_for_phase(model, staged_filter_training, filter_phase_active)
        total_loss_epoch = []
        epoch_time = time.time()
        for batch_x, batch_y in tqdm(train_loader,disable=True):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if args.exclude_experiment and args.exclude_channels:
                batch_x, batch_y = filter_channels(batch_x, batch_y, args.exclude_channels)

            if args.num_submodels > 1:
                start_idx_in = args.current_submodel_idx * args.sub_enc_in
                end_idx_in = (args.current_submodel_idx + 1) * args.sub_enc_in
                batch_x = batch_x[..., start_idx_in:end_idx_in]

                start_idx_out = args.current_submodel_idx * args.sub_c_out
                end_idx_out = (args.current_submodel_idx + 1) * args.sub_c_out
                batch_y = batch_y[..., start_idx_out:end_idx_out]
            outputs = model(batch_x)
            outputs = outputs[:,-args.pred_len:,:]
            batch_y = batch_y[:,-args.pred_len:,:]

            loss = criterion(outputs,batch_y)
            loss.backward()
            optimizer.step()
            total_loss_epoch.append(loss.item())

        # Aggregate and print train loss
        avg_epoch_loss = np.mean(total_loss_epoch)
        if args.distributed:
            loss_tensor = torch.tensor(avg_epoch_loss).to(device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_epoch_loss = loss_tensor.item() / args.world_size
        
        if is_main_process():
            print('Epoch: {} | Train Loss: {:.4f} | Time: {:.4f}s'.format(epoch, avg_epoch_loss, time.time() - epoch_time))

        vali_loss = vali(args, val_data, model, val_loader, criterion, device)
        
        if is_main_process():
            print('Epoch: {} | Validation Loss: {:.4f} '.format(epoch, vali_loss))
            if staged_filter_training and not filter_phase_active:
                warmup_early_stopping(vali_loss,model,args.path)
                should_activate_filter_phase = warmup_early_stopping.early_stop or (epoch + 1) >= filter_warmup_epochs
                if should_activate_filter_phase:
                    reason = "warmup early stopping" if warmup_early_stopping.early_stop else "configured warmup epochs reached"
                    print(f"Activating spectral filter-only phase ({reason}).")
            else:
                early_stopping(vali_loss,model,args.path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if staged_filter_training and not filter_phase_active:
            should_activate_filter_phase = False
            if is_main_process():
                should_activate_filter_phase = warmup_early_stopping.early_stop or (epoch + 1) >= filter_warmup_epochs

            if args.distributed:
                phase_signal = torch.tensor(1.0 if should_activate_filter_phase else 0.0, device=device)
                dist.broadcast(phase_signal, src=0)
                should_activate_filter_phase = phase_signal.item() == 1.0

            if should_activate_filter_phase:
                if is_main_process():
                    load_best_checkpoint(args, model, device)
                if args.distributed:
                    synchronize()
                    load_best_checkpoint(args, model, device)
                set_filter_only_phase(model)
                optimizer, scheduler = build_optimizer_and_scheduler(args, model)
                filter_phase_active = True
                if is_main_process():
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Spectral filter-only phase trainable parameters: {trainable_params/1e6:.2f} M")
        

        scheduler.step()

    if is_main_process():
        load_best_checkpoint(args, model, device)
    
    #if args.distributed:
    #    dist.barrier() # Ensure rank 0 has loaded the model before other ranks proceed or training ends

    return model # Return the model (DDP wrapped if distributed)

def test(args,models,criterion, device):
    test_data,test_loader = get_data(args,flag='test')
    
    sub_models_list = []
    is_ensemble = isinstance(models, list) and len(models) > 1
    if is_ensemble:

        if is_main_process():
            print('loading model for testing')
            original_enc_in = args.enc_in
            original_c_out = args.c_out
            args.enc_in = args.sub_enc_in # for instantiating submodels
            args.c_out = args.sub_c_out
            for i,path in enumerate(models):
                sub_m = Model(args).to(device)
                state_dict = torch.load(path, map_location=device)
                sub_m.load_state_dict(state_dict)
                sub_m.eval()
                sub_models_list.append(sub_m)
            args.enc_in = original_enc_in
            args.c_out = original_c_out
    else:
        model = models
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(torch.load(os.path.join(args.path, 'checkpoint.pth'), map_location=device))
        model.eval()

    preds_list_local = []
    trues_list_local = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if args.exclude_experiment and args.exclude_channels:
                batch_x, batch_y = filter_channels(batch_x, batch_y, args.exclude_channels)
            if is_ensemble:
                all_sub_outputs = []
                for sub_idx, sub_m_instance in enumerate(sub_models_list):
                    start_idx_in = sub_idx * args.sub_enc_in
                    end_idx_in = (sub_idx + 1) * args.sub_enc_in
                    batch_x_sub = batch_x[..., start_idx_in:end_idx_in]

                    sub_output = sub_m_instance(batch_x_sub)
                    all_sub_outputs.append(sub_output[:,-args.pred_len:,:]) 
                outputs = torch.cat(all_sub_outputs, dim=-1) 
            else:
                if args.num_submodels > 1 and hasattr(args, 'current_submodel_idx'):
                    start_idx_in = args.current_submodel_idx * args.sub_enc_in
                    end_idx_in = (args.current_submodel_idx + 1) * args.sub_enc_in
                    batch_x = batch_x[..., start_idx_in:end_idx_in]

                    start_idx_out = args.current_submodel_idx * args.sub_c_out
                    end_idx_out = (args.current_submodel_idx + 1) * args.sub_c_out
                    batch_y = batch_y[..., start_idx_out:end_idx_out]
                outputs = model(batch_x)
                
            
            outputs = outputs[:,-args.pred_len:,:]
            batch_y = batch_y[:,-args.pred_len:,:]
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            preds_list_local.append(pred)
            trues_list_local.append(true)

    preds_local_np = torch.cat(preds_list_local, dim=0).numpy()
    trues_local_np = torch.cat(trues_list_local, dim=0).numpy()

    
    preds = preds_local_np
    trues = trues_local_np

    # Following operations only on rank 0
    if is_main_process():
        test_results_folder_name = 'test_results_ensemble' if is_ensemble else 'test_results_single'
        if args.num_submodels > 1:
            test_results_folder_name += f'test_results_submodel_{args.current_submodel_idx}'
        folder_path = os.path.join(args.path, test_results_folder_name) 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
            
        mae,mse,rmse,mape,mspe = metrics(preds,trues)
        print('MAE: {:.4f} | MSE: {:.4f} | RMSE: {:.4f} | MAPE: {:.4f} | MSPE: {:.4f}'.format(mae, mse, rmse, mape, mspe))
        np.save(os.path.join(folder_path, f'metrics_{args.pred_len}.npy'), np.array([mae, mse, rmse, mape, mspe]))
        return mse,mae # Return MSE for this run
    
    return 0.0 


if __name__ == '__main__':
    import argparse
    import random
    from spectcaster import Model 
    parser = argparse.ArgumentParser(description='Spectcaster')
    # DDP arguments
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training. Default -1 for non-distributed.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')


    # ensemble argument
    parser.add_argument('--num_submodels', type=int, default=1, help='Number of submodels in ensemble')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.txt', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--dates', action='store_false', help='Whether the dataset has dates in the first column')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        # forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model
    parser.add_argument('--revin', dest='revin', action='store_true', help='enable RevIN')
    parser.add_argument('--no-revin', dest='revin', action='store_false', help='disable RevIN')
    parser.set_defaults(revin=True)
    parser.add_argument('--affine', dest='affine', action='store_true', help='use affine transformation in RevIN')
    parser.add_argument('--no-affine', dest='affine', action='store_false', help='do not use affine transformation in RevIN')
    parser.set_defaults(affine=True)
    parser.add_argument('--subtract_last', action='store_true', help='subtract last point in RevIN')
    parser.set_defaults(subtract_last=False)
    parser.add_argument('--individual', dest='individual', action='store_true', help='individual head for each channel')
    parser.add_argument('--no-individual', dest='individual', action='store_false', help='shared head across channels')
    parser.set_defaults(individual=False)
    parser.add_argument('--head_dropout', type=float, default=0.0, help='dropout in the head')
    parser.add_argument('--pos_embed', type=str, default='zeros', help="type of positional embedding (e.g., 'zeros', 'normal')")
    parser.add_argument('--learn_pos_embed', dest='learn_pos_embed', action='store_true', help='make positional embeddings learnable')
    parser.add_argument('--no-learn_pos_embed', dest='learn_pos_embed', action='store_false', help='keep positional embeddings fixed')
    parser.set_defaults(learn_pos_embed=True)

    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers for iTransformer')
    parser.add_argument('--enc_in', type=int, default=862, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=862, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=862, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--patch_len', type = int, default=16, help='')
    parser.add_argument('--depth', type = int, default=5, help='')
    parser.add_argument('--alpha', type=int, default=2, help='Number of spectral filtering blocks')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)

        # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='StepLR step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.9, help='StepLR gamma')
    parser.add_argument('--use_norm', action='store_false', help='Use normalization layers in the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=int, default=30,help='Patience for early stopping')  
    parser.add_argument('--filter_warmup_epochs', type=int, default=0,
                    help='If > 0, bypass and freeze spectral filters while training the transformer, then randomize filters and train only filters.')
    parser.add_argument('--exclude_channels', type=int, nargs='*', default=[], 
                    help='List of channel indices to exclude from training/testing (0-indexed)')
    parser.add_argument('--exclude_experiment', action='store_true', 
                    help='Enable channel exclusion experiment mode')
    args = parser.parse_args()

    args.original_enc_in = args.enc_in
    args.original_c_out = args.c_out

    if args.exclude_experiment and args.exclude_channels:
        if is_main_process():
            print(f"Excluding channels: {args.exclude_channels}")
            print(f"Original channels: {args.original_enc_in}")
        excluded_count = len(args.exclude_channels)
        args.enc_in =args.original_enc_in - excluded_count
        args.c_out = args.original_c_out - excluded_count
    
        if is_main_process():
            print(f"Adjusted channels: {args.enc_in}")

    if args.num_submodels > 1:
        if args.enc_in % args.num_submodels != 0:
            raise ValueError(f"enc_in {args.original_enc_in} must be divisible by num_submodels {args.num_submodels}")
        if args.c_out % args.num_submodels != 0:
            raise ValueError(f"c_out {args.original_c_out} must be divisible by num_submodels {args.num_submodels}")
        args.sub_enc_in = args.enc_in // args.num_submodels
        args.sub_c_out = args.c_out // args.num_submodels
    else:
        args.sub_enc_in = args.enc_in
        args.sub_c_out = args.c_out
    # Initialize DDP
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if args.local_rank != -1 and torch.cuda.is_available():
            
            args.device = f'cuda:{args.local_rank}'
            torch.cuda.set_device(args.local_rank)
        else:
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # +++ Initialize distributed environment
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1
        args.distributed = False

    if args.distributed:
        print(f"Initializing process group for rank {args.local_rank}...")
        dist.init_process_group(backend='nccl', init_method='env://')
        # Ensure setup_for_distributed is called after init_process_group
        setup_for_distributed(args.local_rank == 0) # Pass is_main_process flag
        print(f"Process group initialized for rank {args.local_rank}.")
    else:
        # Still call setup for non-distributed case to set the flag
        setup_for_distributed(True) 

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA because this script uses the NCCL backend.")
        args.gpu = args.local_rank
        device = torch.device('cuda', args.gpu)
    else:
        args.gpu = 0
        device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    
    fix_seed = 42 
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    

    prediction_lengths = [96,192,336,720] 
    all_run_metrics = {} 

    for pred_len in prediction_lengths:
        if is_main_process():
            print(f"\n>>>>>>> Processing Prediction Length: {pred_len} <<<<<<<")
        args.pred_len = pred_len

        submodel_checkpoint_paths = []
        if args.num_submodels > 1:
            for i in range(args.num_submodels):
                args.current_submodel_idx = i
                args.enc_in = args.sub_enc_in
                args.c_out = args.sub_c_out

                if is_main_process():
                    print(f"Setting up submodel {i+1}/{args.num_submodels} -- ")

                setting = f'{args.data}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}_patch{args.patch_len}_depth{args.depth}_drop{args.dropout}_ind{args.individual}_rev{args.revin}_lr{args.learning_rate}_bs{args.batch_size}_submodel{i}'
                if args.filter_warmup_epochs > 0:
                    setting += f'_filterwarm{args.filter_warmup_epochs}'
                args.path = os.path.join(args.checkpoints, setting) # Path for this run's checkpoints
        
                if is_main_process():
                    print(f"Setting: {setting}")
                    if not os.path.exists(args.path):
                        os.makedirs(args.path)
        

                current_submodel = Model(args).to(args.device)
        
                if is_main_process(): # Calculate and print params only on rank 0
                    total_params = sum(p.numel() for p in current_submodel.parameters() if p.requires_grad)
                    print(f"Total trainable parameters: {total_params/1e6:.2f} M")

                if args.distributed:
                    nn.SyncBatchNorm.convert_sync_batchnorm(current_submodel) # Convert to SyncBatchNorm if needed
                    print(args.local_rank)
                    current_submodel = DDP(current_submodel, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False) # Set find_unused_parameters based on your model

                criterion = nn.MSELoss() 
                optimizer = optim.Adam(current_submodel.parameters(), lr=args.learning_rate)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        
                if is_main_process():
                    print(">>>>>>> Start Training <<<<<<<<")
                args.dates = True if args.data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2','Weather'] else False # Set dates based on dataset
                trained_model = train(args, current_submodel, criterion, optimizer, args.device, scheduler)
                if is_main_process():
                    checkpoint_path = os.path.join(args.path, 'checkpoint.pth')
                    if os.path.exists(checkpoint_path):
                        submodel_checkpoint_paths.append(checkpoint_path)

            # training done
            if is_main_process():
                print(">>>>>>> Start Testing <<<<<<<<")
            args.enc_in = args.original_enc_in
            args.c_out = args.original_c_out

            if is_main_process():
                print("\n >>>>>>>> Ensemble Testing <<<<<<<<<" if args.num_submodels > 1 else "\n >>>>>>>> Single Model Testing <<<<<<<<<")
                if len(submodel_checkpoint_paths) == args.num_submodels:
                    current_mse = test(args, submodel_checkpoint_paths, criterion, args.device) # test returns mse for rank 0, 0.0 for others
        
            if is_main_process():
                all_run_metrics[pred_len] = current_mse # Store only MSE, or a tuple/dict of all metrics
        else: # Single model case
            setting = f'{args.data}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}_patch{args.patch_len}_depth{args.depth}_drop{args.dropout}_ind{args.individual}_rev{args.revin}_lr{args.learning_rate}_bs{args.batch_size}'
            if args.filter_warmup_epochs > 0:
                setting += f'_filterwarm{args.filter_warmup_epochs}'
            args.path = os.path.join(args.checkpoints, setting)
            if is_main_process():
                print(f"Setting: {setting}")
                if not os.path.exists(args.path):
                    os.makedirs(args.path)
            current_model = Model(args).to(args.device)
            if is_main_process():
                total_params = sum(p.numel() for p in current_model.parameters() if p.requires_grad)
                print(f"Total trainable parameters: {total_params/1e6:.2f} M")
            
            if args.distributed:
                current_model = nn.SyncBatchNorm.convert_sync_batchnorm(current_model) # Convert to SyncBatchNorm if needed
                current_model = DDP(current_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(current_model.parameters(), lr=args.learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

            if is_main_process():
                print(">>>>>>> Start Training <<<<<<<<")
            args.dates = True if args.data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2','Weather'] else False
            trained_model = train(args, current_model, criterion, optimizer, args.device, scheduler)

            if is_main_process():
                print(">>>>>>> Start Testing <<<<<<<<")
                current_mse, current_mae = test(args, trained_model, criterion, args.device)
                all_run_metrics[pred_len] = (current_mse,current_mae)
    if is_main_process():
        print("\n\n======== Summary of All Runs ========")
        for pred_len, metrics_tuple in all_run_metrics.items():
            mse_val, mae_val = metrics_tuple 
            print(f"Prediction Length: {pred_len}")
            print(f"  MSE: {mse_val:.4f} | MAE: {mae_val:.4f}") # Assuming current_metrics was just MSE

    if args.distributed:
        dist.destroy_process_group() 
        sys.exit(0) # Exit cleanly after distributed training
        
