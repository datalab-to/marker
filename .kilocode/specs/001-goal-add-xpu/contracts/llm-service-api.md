# LLM Service API Contract

## Overview
This contract defines the new Llama-CPP LLM service implementation that integrates with the llama-serve backend.

## Service Interface

### Class: LlamaCPPService
Implements the BaseService interface for Llama-CPP integration.

#### Constructor
```python
LlamaCPPService(config: Optional[BaseModel | dict] = None)
```

#### Configuration Parameters
- **llama_cpp_base_url**: String - Base URL for the llama-serve instance
- **llama_cpp_model**: String - Model name to use for inference
- **timeout**: Integer - Request timeout in seconds (default: 30)
- **max_retries**: Integer - Maximum number of retry attempts (default: 2)

#### Method: __call__
Executes an LLM inference request.

```python
def __call__(
    self,
    prompt: str,
    image: PIL.Image.Image | List[PIL.Image.Image] | None,
    block: Block | None,
    response_schema: type[BaseModel],
    max_retries: int | None = None,
    timeout: int | None = None,
)
```

##### Parameters
- **prompt**: String - The prompt to send to the LLM
- **image**: PIL Image or List of PIL Images - Optional images to include
- **block**: Block - The document block being processed
- **response_schema**: Pydantic BaseModel - Expected response format
- **max_retries**: Integer - Override default max retries
- **timeout**: Integer - Override default timeout

##### Returns
Dictionary containing the LLM response parsed according to response_schema.

## API Endpoints (llama-serve)

### POST /api/generate
Generates a response from the LLM.

#### Request
```json
{
  "model": "llama3.2-vision",
  "prompt": "Describe this document section",
  "stream": false,
  "format": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "summary": {"type": "string"}
    },
    "required": ["title", "summary"]
  },
  "images": ["base64_encoded_image_data"]
}
```

#### Response
```json
{
  "model": "llama3.2-vision",
  "response": "{\"title\": \"Introduction\", \"summary\": \"This section provides an overview of the document\"}",
  "prompt_eval_count": 45,
  "eval_count": 23
}
```

## Integration Points
- Integrated with `marker/services/__init__.py` BaseService
- Used by LLM processors in `marker/processors/llm/`
- Configured through `marker/config/parser.py` CLI options
- Selected via `--llm_service marker.services.llama_cpp.LlamaCPPService` option

## Backward Compatibility
- All existing LLM services continue to work as before
- New service follows the same interface as existing services
- Configuration method is consistent with existing services

## Phase 3.1 Completion Status
Phase 3.1 has been completed with the successful implementation of:
- Llama-CPP LLM service API contract
- Integration with marker/services/__init__.py BaseService
- Backward compatibility maintained for all existing LLM services
- All integration points successfully implemented and tested