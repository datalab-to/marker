# Model-Level DDP Process Creation Design

## Overview
This document outlines the design for implementing model-level DDP process creation based on worker parameters. The goal is to create actual DDP processes for parallel execution of a single model when invoked with a specified number of workers, rather than using the workers parameter for internal threading as in the CUDA implementation.

## Current Implementation Analysis
The current implementation uses:
- Worker processes for parallelizing multiple documents
- Internal threading for model execution within each worker
- DDP for multi-GPU setups at the document level

We need to add:
- Model-level DDP processes for parallelizing single document processing
- Dynamic DDP process creation based on worker parameters
- Proper coordination between main process and DDP workers

## Design Components

### 1. ModelDDPProxy Class
A proxy class that intercepts model calls and creates DDP processes when needed.

```python
class ModelDDPProxy:
    def __init__(self, model_class, workers, *args, **kwargs):
        self.workers = workers
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, *args, **kwargs):
        if self.workers > 1 and should_use_ddp():
            return launch_model_ddp_processes(
                self.model_class, self.workers, 
                self.args, self.kwargs, args, kwargs)
        else:
            # Fall back to normal model execution
            model = self.model_class(*self.args, **self.kwargs)
            return model(*args, **kwargs)
```

### 2. launch_model_ddp_processes Function
Function to dynamically launch DDP processes based on worker count.

```python
def launch_model_ddp_processes(model_class, worker_count, model_args, model_kwargs, call_args, call_kwargs):
    # Launch N DDP processes based on worker_count
    # Set up proper environment variables
    # Distribute model_args/model_kwargs to processes
    # Collect and aggregate results
    pass
```

### 3. Enhanced wrap_model_with_ddp Function
Modified function to recognize when to create multiple processes.

```python
def wrap_model_with_ddp(model, device=None, workers=None):
    # If workers > 1 and DDP enabled, create DDP processes
    # Otherwise use existing wrapping logic
    pass
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create ModelDDPProxy class
2. Implement launch_model_ddp_processes function
3. Enhance wrap_model_with_ddp function

### Phase 2: Process Management
1. Create process launcher with proper environment variable setup
2. Implement data distribution and result aggregation
3. Add error handling for process failures

### Phase 3: Integration
1. Modify model constructors to accept worker parameters
2. Update existing models to utilize the new mechanism
3. Add coordination logic between main process and DDP workers

### Phase 4: Optimization
1. Add VRAM-aware process allocation
2. Implement process pooling to reduce startup overhead
3. Add batching mechanisms for better throughput

## Technical Details

### Process Launching
- Use `torch.multiprocessing` for process creation
- Set up proper environment variables for DDP
- Use ccl backend for XPU devices

### Data Distribution
- Serialize model inputs/outputs for inter-process communication
- Implement efficient data transfer mechanisms
- Handle different data types (tensors, lists, dicts, etc.)

### Result Aggregation
- Collect results from all DDP processes
- Merge results according to model-specific logic
- Handle errors and partial failures

## Integration Points

### Model Constructors
Models need to be modified to accept and handle worker parameters:
```python
class LayoutPredictor:
    def __init__(self, device=None, dtype=None, workers=None):
        self.device = device
        self.dtype = dtype
        self.workers = workers # New parameter
        # ... rest of initialization
```

### Configuration
Add worker parameter handling in the configuration system:
- CLI options for worker count per model
- Config file support for worker settings
- Environment variable support

## Performance Considerations

### VRAM Management
- Implement VRAM-aware process allocation
- Monitor memory usage to prevent OOM errors
- Balance load across available XPU devices

### Process Pooling
- Reuse DDP processes to reduce startup overhead
- Implement process lifecycle management
- Handle process failures and restarts

### Batching
- Implement batching mechanisms for better throughput
- Optimize batch sizes based on VRAM availability
- Coordinate batching across DDP processes

## Error Handling

### Process Failures
- Detect and handle process crashes
- Implement retry mechanisms
- Graceful degradation to single-process execution

### Communication Errors
- Handle inter-process communication failures
- Implement timeouts for process coordination
- Provide meaningful error messages

## Testing Strategy

### Unit Tests
- Test ModelDDPProxy functionality
- Verify process launching and cleanup
- Validate data distribution and aggregation

### Integration Tests
- Test with actual models and real data
- Verify performance improvements
- Check resource utilization

### Edge Cases
- Test with different worker counts
- Verify behavior with single worker
- Test error conditions and recovery

## Documentation

### Usage Guide
- How to enable model-level DDP
- Configuration options and parameters
- Performance tuning guidelines

### API Documentation
- ModelDDPProxy class documentation
- launch_model_ddp_processes function documentation
- Integration with existing model APIs