from typing import Any, Dict


def call_model_sync(model_name: str, model_dict: Dict[str, Any], *args, **kwargs):
    """
    Call a model synchronously, using batcher if available, otherwise direct call.
    
    Args:
        model_name: Name of the model in the model_dict
        model_dict: Dictionary containing models and potentially batchers
        *args: Arguments to pass to the model
        **kwargs: Keyword arguments to pass to the model
        
    Returns:
        Result of the model call
    """
    # Handle None model_dict
    if model_dict is None:
        model_dict = {}
    
    # Check if we have batchers and this model has a batcher
    if "_batchers" in model_dict and model_name in model_dict["_batchers"]:
        # Use batcher
        batcher = model_dict["_batchers"][model_name]
        # For now, we'll create a simple item that contains both args and kwargs
        # In a real implementation, this would need to be more sophisticated
        item = {"args": args, "kwargs": kwargs}
        return batcher.submit_sync(item)
    else:
        # Direct model call
        if model_name not in model_dict:
            raise KeyError(f"Model '{model_name}' not found in model dictionary. Available models: {list(model_dict.keys())}")
        model = model_dict[model_name]
        return model(*args, **kwargs)