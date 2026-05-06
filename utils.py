import torch.distributed as dist
import builtins as __builtin__
import torch


_is_main_process = True

def filter_channels(batch_x, batch_y, exclude_indices):
    """
    For use on channel withholding experiments
    """
    if not exclude_indices:
        return batch_x, batch_y
    
    # Create mask for channels to keep
    total_channels = batch_x.shape[-1]
    keep_mask = torch.ones(total_channels, dtype=torch.bool, device=batch_x.device)
    keep_mask[exclude_indices] = False
    
    # Filter channels
    batch_x_filtered = batch_x[..., keep_mask]
    batch_y_filtered = batch_y[..., keep_mask]
    
    return batch_x_filtered, batch_y_filtered

def setup_for_distributed(is_main_process_arg):
	global _is_main_process
	_is_main_process = is_main_process_arg
	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop('force',False)
		if _is_main_process or force:
			builtin_print(*args, **kwargs)
	
	__builtin__.print = print

def is_main_process():
	return _is_main_process

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
