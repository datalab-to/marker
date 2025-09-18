# Model-Level DDP Process Creation Implementation Plan

## Overview
This document provides a detailed implementation plan for adding model-level DDP process creation based on worker parameters. The implementation will allow models to be executed in parallel DDP processes when a worker count greater than 1 is specified.

## Implementation Phases

### Phase 1: Core Infrastructure

#### 1.1 Enhanced wrap_model_with_ddp Function
Modify the existing `wrap_model_with_ddp` function in `marker/models.py`:

```python
def wrap_model_with_ddp(model, device=None, workers=None):
    """
    Wrap model with DDP if conditions are met, or create DDP processes based on worker count.
    
    Args:
        model: The model to wrap
        device: The device to use (if None, will be determined from DDP rank)
        workers: Number of worker processes to create for model-level parallelism
        
    Returns:
        The model (wrapped with DDP if applicable, or proxied for DDP process creation)
    """
    if not DDP_AVAILABLE:
        return model
        
    # If workers > 1, create a proxy for DDP process creation
    if workers and workers > 1 and should_use_model_ddp():
        from marker.models_ddp import ModelDDPProxy
        return ModelDDPProxy(model.__class__, workers, device=device)
        
    try:
        import torch.distributed as dist
        import torch.nn as nn
        
        # Check if DDP is initialized
        if not dist.is_initialized():
            return model
            
        # Get device for this process
        if device is None:
            # For single XPU device, all processes use the same device
            # The device index should be 0 for single device scenarios
            device = "xpu:0"
            
        # Move model to device
        model = model.to(device)
        
        # Wrap with DDP if we have multiple processes
        if dist.get_world_size() > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
            logger.debug(f"Model wrapped with DDP on device {device}")
            
        return model
    except Exception as e:
        logger.warning(f"Failed to wrap model with DDP: {e}")
        return model
```

#### 1.2 ModelDDPProxy Class
Create `marker/models_ddp.py` with the ModelDDPProxy class:

```python
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from marker.logger import get_logger

logger = get_logger()

class ModelDDPProxy:
    def __init__(self, model_class, workers, *args, **kwargs):
        self.model_class = model_class
        self.workers = workers
        self.args = args
        self.kwargs = kwargs
        self.device = kwargs.get('device', None)
        
    def __call__(self, *args, **kwargs):
        if self.workers > 1 and should_use_model_ddp():
            return launch_model_ddp_processes(
                self.model_class, self.workers, 
                self.args, self.kwargs, args, kwargs)
        else:
            # Fall back to normal model execution
            model = self.model_class(*self.args, **self.kwargs)
            return model(*args, **kwargs)

def should_use_model_ddp():
    """Determine if model-level DDP should be used."""
    return (os.environ.get('USE_MODEL_DDP', 'false').lower() == 'true' or
            os.environ.get('USE_XPU_DDP', 'false').lower() == 'true')

def launch_model_ddp_processes(model_class, worker_count, model_args, model_kwargs, call_args, call_kwargs):
    """
    Launch DDP processes for model execution and collect results.
    
    Args:
        model_class: The model class to instantiate
        worker_count: Number of DDP processes to create
        model_args: Arguments for model constructor
        model_kwargs: Keyword arguments for model constructor
        call_args: Arguments for model call
        call_kwargs: Keyword arguments for model call
        
    Returns:
        Aggregated results from all DDP processes
    """
    # Create a multiprocessing manager for shared data
    manager = mp.Manager()
    result_dict = manager.dict()
    
    # Create processes
    processes = []
    for rank in range(worker_count):
        p = mp.Process(
            target=model_ddp_worker,
            args=(rank, worker_count, model_class, model_args, model_kwargs, 
                  call_args, call_kwargs, result_dict)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Aggregate results (this will depend on the specific model)
    return aggregate_results(result_dict)

def model_ddp_worker(rank, world_size, model_class, model_args, model_kwargs, 
                     call_args, call_kwargs, result_dict):
    """
    Worker function for DDP model execution.
    """
    try:
        # Set up DDP environment for this worker
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'  # Different port than document-level DDP
        
        # Initialize the process group
        dist.init_process_group(backend='ccl', rank=rank, world_size=world_size)
        
        # Create model instance
        model = model_class(*model_args, **model_kwargs)
        
        # Move model to device if specified
        if model_kwargs.get('device'):
            model = model.to(model_kwargs['device'])
        
        # Wrap with DDP
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[model_kwargs.get('device', 'xpu:0')] if model_kwargs.get('device') else None
            )
        
        # Execute model with provided arguments
        result = model(*call_args, **call_kwargs)
        
        # Store result
        result_dict[rank] = result
        
        # Clean up
        dist.destroy_process_group()
        
    except Exception as e:
        logger.error(f"Error in DDP worker {rank}: {e}")
        result_dict[rank] = None

def aggregate_results(result_dict):
    """
    Aggregate results from multiple DDP processes.
    This is a placeholder - actual implementation will depend on the model type.
    """
    # For now, just return the first result
    # In practice, this would need to be model-specific
    if 0 in result_dict:
        return result_dict[0]
    return None
```

### Phase 2: Process Management

#### 2.1 Data Distribution
Implement data distribution mechanisms for different model types:

```python
def distribute_data_for_model(model_class, data, worker_count):
    """
    Distribute data across workers based on model type.
    """
    # This would need to be implemented for each model type
    # For example, for batch processing models, split the batch
    # For models that process lists, split the list
    pass

def collect_and_merge_results(results):
    """
    Collect and merge results from multiple workers.
    """
    # This would need to be implemented for each model type
    # For example, for batch processing models, concatenate batches
    # For models that return lists, merge the lists
    pass
```

#### 2.2 Error Handling
Implement robust error handling:

```python
def handle_worker_failure(rank, error):
    """
    Handle worker process failures.
    """
    logger.error(f"Worker {rank} failed with error: {error}")
    # Implement retry logic or fallback to single process execution
```

### Phase 3: Integration with Existing Models

#### 3.1 Modify create_model_dict
Update the `create_model_dict` function to accept worker parameters:

```python
def create_model_dict(
    device=None, dtype=None, attention_implementation: str | None = None,
    wrap_with_ddp: bool = False,
    model_workers: dict = None  # New parameter: {'layout_model': 2, 'recognition_model': 4, ...}
) -> dict:
    if model_workers is None:
        model_workers = {}
    
    foundation_predictor = FoundationPredictor(
        device=device, dtype=dtype, attention_implementation=attention_implementation
    )
    models = {
        "foundation_model": foundation_predictor,
        "layout_model": LayoutPredictor(device=device, dtype=dtype),
        "recognition_model": RecognitionPredictor(foundation_predictor),
        "table_rec_model": TableRecPredictor(device=device, dtype=dtype),
        "detection_model": DetectionPredictor(device=device, dtype=dtype),
        "ocr_error_model": OCRErrorPredictor(device=device, dtype=dtype),
    }
    
    # Optionally wrap models with DDP or create DDP proxies
    if wrap_with_ddp or model_workers:
        for name, model in models.items():
            workers = model_workers.get(name, None)
            models[name] = wrap_model_with_ddp(model, device, workers)
    
    return models
```

#### 3.2 Update Model Constructors
Since we can't modify the surya models directly, we'll need to create wrapper classes or use proxy patterns.

### Phase 4: Configuration and CLI Integration

#### 4.1 Add CLI Options
Add worker parameters to the CLI configuration:

```python
# In marker/config/parser.py
def common_options(fn):
    # ... existing options ...
    
    fn = click.option(
        "--layout_workers",
        type=int,
        default=1,
        help="Number of workers for layout model DDP processes",
    )(fn)
    fn = click.option(
        "--recognition_workers",
        type=int,
        default=1,
        help="Number of workers for recognition model DDP processes",
    )(fn)
    # ... add similar options for other models ...
    
    return fn
```

#### 4.2 Process Worker Configuration
Update the configuration processing:

```python
# In marker/config/parser.py
def generate_config_dict(self) -> Dict[str, Any]:
    config = {}
    # ... existing config processing ...
    
    # Add model worker configuration
    model_workers = {}
    if self.cli_options.get("layout_workers", 1) > 1:
        model_workers["layout_model"] = self.cli_options["layout_workers"]
    if self.cli_options.get("recognition_workers", 1) > 1:
        model_workers["recognition_model"] = self.cli_options["recognition_workers"]
    # ... add similar for other models ...
    
    if model_workers:
        config["model_workers"] = model_workers
    
    return config
```

### Phase 5: Performance Optimization

#### 5.1 VRAM Management
Implement VRAM-aware process allocation:

```python
def calculate_optimal_workers(model_type, available_vram, model_vram_per_worker):
    """
    Calculate optimal number of workers based on available VRAM.
    """
    max_workers = max(1, available_vram // model_vram_per_worker)
    return max_workers

def get_model_vram_requirements(model_class):
    """
    Get VRAM requirements for a model type.
    This would need to be implemented based on empirical data.
    """
    # Return estimated VRAM in GB per worker
    model_vram_map = {
        'LayoutPredictor': 3,
        'RecognitionPredictor': 4,
        'TableRecPredictor': 2,
        # ... add other models ...
    }
    return model_vram_map.get(model_class.__name__, 2)
```

#### 5.2 Process Pooling
Implement process pooling to reduce startup overhead:

```python
class DDPProcessPool:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.process_pool = []
        self.available_workers = []
        
    def get_worker(self, model_class, model_args, model_kwargs):
        """
        Get an available worker or create a new one.
        """
        # Implementation for process pooling
        pass
        
    def release_worker(self, worker):
        """
        Return a worker to the pool.
        """
        # Implementation for process pooling
        pass
```

## Integration Points

### Worker Initialization
Update the worker initialization in `marker/scripts/convert.py`:

```python
def worker_init():
    # Check if DDP should be used
    use_ddp = False
    try:
        # Check if we're in an XPU DDP environment
        import torch.distributed as dist
        import torch
        if (hasattr(torch, 'xpu') and torch.xpu.is_available() and
            os.environ.get('USE_XPU_DDP', 'false').lower() == 'true' and
            dist.is_available()):
            use_ddp = True
    except ImportError:
        pass
    
    # Create models with optional DDP wrapping
    # Pass model_workers from global config
    model_workers = getattr(worker_init, 'model_workers', {})
    model_dict = create_model_dict(wrap_with_ddp=use_ddp, model_workers=model_workers)
    
    global model_refs
    model_refs = model_dict
    
    # Ensure we clean up the model references on exit
    atexit.register(worker_exit)
```

## Testing Strategy

### Unit Tests
Create unit tests for the new functionality:

```python
# tests/test_models_ddp.py
import unittest
from marker.models_ddp import ModelDDPProxy, should_use_model_ddp

class TestModelDDP(unittest.TestCase):
    def test_proxy_creation(self):
        """Test that ModelDDPProxy is created correctly."""
        # Test implementation
        
    def test_ddp_decision(self):
        """Test DDP usage decision logic."""
        # Test implementation
```

### Integration Tests
Create integration tests with actual models:

```python
# tests/integration/test_model_ddp_integration.py
import unittest
from marker.models import create_model_dict

class TestModelDDPIntegration(unittest.TestCase):
    def test_model_with_workers(self):
        """Test model creation with worker parameters."""
        # Test implementation
```

## Documentation

### Usage Guide
Document how to use the new functionality:

```markdown
# Model-Level DDP Usage

To enable model-level DDP processing, use the following CLI options:

```bash
marker --layout_workers 2 --recognition_workers 4 input.pdf
```

This will create 2 DDP processes for the layout model and 4 DDP processes for the recognition model.

### Environment Variables

- `USE_MODEL_DDP`: Set to "true" to enable model-level DDP
- `USE_XPU_DDP`: Set to "true" to enable XPU DDP (also enables model-level DDP)
```

## Implementation Order

1. Create `marker/models_ddp.py` with core functionality
2. Modify `marker/models.py` to support worker parameters
3. Update CLI configuration in `marker/config/parser.py`
4. Update worker initialization in `marker/scripts/convert.py`
5. Add performance optimizations
6. Create tests
7. Document usage