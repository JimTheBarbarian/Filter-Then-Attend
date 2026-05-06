import torch
import torch.nn as nn
import time
import argparse
import numpy as np
from fvcore.nn import FlopCountAnalysis
from spectcaster import Model
from iTransformer import iTransformer
from PaiFilter import PaiFilter
from Leddam import Leddam_model
import warnings
warnings.filterwarnings('ignore')

def create_input_tensor(args):
    """Create input tensor based on configuration"""
    batch_size = args.batch_size
    seq_len = args.seq_len
    enc_in = args.enc_in
    
    # Create dummy input tensor
    input_tensor = torch.randn(batch_size, seq_len, enc_in)
    return input_tensor

def count_flops(model, input_tensor, device='cuda'):
    """Count FLOPs using fvcore"""
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    
    return total_flops

def measure_training_time(model, input_tensor, target_tensor, criterion, optimizer, device='cuda    ', num_iterations=100):
    """Measure training time per iteration"""
    model.train()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    # Warmup iterations
    for _ in range(10):
        optimizer.zero_grad()
        output = model(input_tensor)
        output = output[:, -target_tensor.size(1):, :]
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
    
    # Synchronize GPU operations
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual timing
    start_time = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        output = output[:, -target_tensor.size(1):, :]
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000
    time_per_iteration_ms = total_time_ms / num_iterations
    
    return time_per_iteration_ms

def measure_inference_time(model, input_tensor, device='cpu', num_iterations=100):
    """Measure inference time per iteration"""
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time_ms = (end_time - start_time) * 1000
    time_per_iteration_ms = total_time_ms / num_iterations
    
    return time_per_iteration_ms

def create_target_tensor(args):
    """Create target tensor for training time measurement"""
    batch_size = args.batch_size
    pred_len = args.pred_len
    c_out = args.c_out
    
    target_tensor = torch.randn(batch_size, pred_len, c_out)
    return target_tensor

def benchmark_model(args, device='cpu', num_timing_iterations=100):
    """Complete benchmark of a model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking Model Configuration:")
    print(f"Data: {args.data}")
    print(f"Seq Length: {args.seq_len}, Pred Length: {args.pred_len}")
    print(f"Model Dim: {args.d_model}, Heads: {args.n_heads}")
    print(f"Encoder Input: {args.enc_in}, Output: {args.c_out}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Create model
    model = Leddam_model(args)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create input tensors
    input_tensor = create_input_tensor(args)
    target_tensor = create_target_tensor(args)
    
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Target Shape: {target_tensor.shape}")
    
    # Count FLOPs
    try:
        total_flops = count_flops(model, input_tensor, device)
        print(f"FLOPs: {total_flops:,} ({total_flops/1e9:.2f}G)")
        flops_per_sample = total_flops / args.batch_size
        print(f"FLOPs per sample: {flops_per_sample:,} ({flops_per_sample/1e9:.2f}G)")
    except Exception as e:
        print(f"FLOP counting failed: {e}")
        total_flops = None
    
    # Measure inference time
    try:
        inference_time = measure_inference_time(model, input_tensor, device, num_timing_iterations)
        print(f"Inference Time: {inference_time:.3f} ms/iteration")
        if total_flops:
            throughput = total_flops / (inference_time / 1000)  # FLOPs per second
            print(f"Inference Throughput: {throughput/1e9:.2f} GFLOP/s")
    except Exception as e:
        print(f"Inference timing failed: {e}")
    
    # Measure training time
    try:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        training_time = measure_training_time(
            model, input_tensor, target_tensor, criterion, optimizer, 
            device, num_timing_iterations
        )
        print(f"Training Time: {training_time:.3f} ms/iteration")
        if total_flops:
            # Training typically requires ~3x more FLOPs than inference (forward + 2x backward)
            training_flops = total_flops * 3
            training_throughput = training_flops / (training_time / 1000)
            print(f"Training Throughput: {training_throughput/1e9:.2f} GFLOP/s")
    except Exception as e:
        print(f"Training timing failed: {e}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': total_flops,
        'inference_time_ms': inference_time if 'inference_time' in locals() else None,
        'training_time_ms': training_time if 'training_time' in locals() else None
    }

def run_comprehensive_benchmark():
    """Run benchmarks across different configurations"""
    parser = argparse.ArgumentParser(description='FLOP and Timing Benchmark')
    
    # Model arguments (copying from exp.py)
    parser.add_argument('--data', type=str, default='Electricity')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of training')
    # model
    parser.add_argument('--revin', action='store_false', help='enable RevIN')
    parser.add_argument('--no-revin', dest='revin', action='store_false', help='disable RevIN')
    parser.set_defaults(revin=True)
    parser.add_argument('--affine', action='store_true', help='use affine transformation in RevIN')
    parser.add_argument('--no-affine', dest='affine', action='store_false', help='do not use affine transformation in RevIN')
    parser.set_defaults(affine=True)
    parser.add_argument('--subtract_last', action='store_true', help='subtract last point in RevIN')
    parser.set_defaults(subtract_last=False)
    parser.add_argument('--individual', action='store_false', help='individual head for each channel')
    parser.set_defaults(individual=False)
    parser.add_argument('--head_dropout', type=float, default=0.0, help='dropout in the head')
    parser.add_argument('--pos_embed', type=str, default='zeros', help="type of positional embedding (e.g., 'zeros', 'normal')")
    parser.add_argument('--learn_pos_embed', action='store_false', help='make positional embeddings learnable')
    parser.add_argument('--no-learn_pos_embed', dest='learn_pos_embed', action='store_false')
    parser.set_defaults(learn_pos_embed=True)

    parser.add_argument('--use_norm', action='store_false', help='Use normalization layers in the model')
    parser.set_defaults(use_norm=True)
    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers for iTransformer')
    parser.add_argument('--enc_in', type=int, default=321, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=321, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=321, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--patch_len', type = int, default=16, help='')
    parser.add_argument('--depth', type = int, default=4, help='')
    parser.add_argument('--alpha', type=int, default=1, help='Number of spectral filtering blocks')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # Benchmark specific arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--timing_iterations', type=int, default=100, help='Number of iterations for timing')
    parser.add_argument('--benchmark_configs', action='store_true', help='Run multiple configurations')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if args.benchmark_configs:
        # Test different configurations
        configs = [
            # Different model sizes
            #{'d_model': 64, 'n_heads': 4, 'd_ff': 64, 'name': 'Small'},
            #{'d_model': 128, 'n_heads': 8, 'd_ff': 128, 'name': 'Medium'},
            {'d_model': 256, 'n_heads': 8, 'd_ff': 256, 'name': 'Large'},
            
            # Different sequence lengths
            {'seq_len': 96, 'name': 'Short Seq'},
            #{'seq_len': 336, 'name': 'Medium Seq'},
            #{'seq_len': 720, 'name': 'Long Seq'},
            
            # Different prediction lengths
            {'pred_len': 96, 'name': 'Pred 96'},
            #{'pred_len': 192, 'name': 'Pred 192'},
            #{'pred_len': 336, 'name': 'Pred 336'},
        ]
        
        results = []
        for config in configs:
            config_args = args
            for key, value in config.items():
                if key != 'name':
                    setattr(config_args, key, value)
            
            print(f"\n🔍 Testing Configuration: {config['name']}")
            result = benchmark_model(config_args, device, args.timing_iterations)
            result['config_name'] = config['name']
            results.append(result)
        
        # Summary
        print(f"\n\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"{'Config':<15} {'Params(M)':<12} {'FLOPs(G)':<12} {'Inf(ms)':<10} {'Train(ms)':<12}")
        print(f"{'-'*80}")
        
        for result in results:
            params_m = result['trainable_params'] / 1e6 if result['trainable_params'] else 0
            flops_g = result['flops'] / 1e9 if result['flops'] else 0
            inf_time = result['inference_time_ms'] if result['inference_time_ms'] else 0
            train_time = result['training_time_ms'] if result['training_time_ms'] else 0
            
            print(f"{result['config_name']:<15} {params_m:<12.2f} {flops_g:<12.2f} {inf_time:<10.2f} {train_time:<12.2f}")
    
    else:
        # Single configuration benchmark
        result = benchmark_model(args, device, args.timing_iterations)

if __name__ == '__main__':
    run_comprehensive_benchmark()