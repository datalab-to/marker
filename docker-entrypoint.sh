#!/bin/bash
set -e

echo "======================================"
echo "Marker PDF Server - Docker Container"
echo "======================================"

# Pre-download models if they don't exist
if [ ! -d "/root/.cache/huggingface/hub" ] || [ -z "$(ls -A /root/.cache/huggingface/hub 2>/dev/null)" ]; then
    echo "First run detected - downloading models..."
    echo "This may take several minutes depending on your internet connection."
    python3 -c "from marker.models import create_model_dict; create_model_dict()" || {
        echo "Warning: Model download may have failed, but continuing anyway..."
    }
    echo "Models downloaded successfully!"
else
    echo "Models already cached, skipping download."
fi

# Build server command with environment variable options
SERVER_CMD="marker_server"

# Basic server configuration
SERVER_CMD="$SERVER_CMD --port ${PORT:-8001}"
SERVER_CMD="$SERVER_CMD --host ${HOST:-0.0.0.0}"

# LLM Service Configuration
if [ -n "$LLM_SERVICE" ]; then
    SERVER_CMD="$SERVER_CMD --llm_service $LLM_SERVICE"
fi

# Use LLM flag
if [ "$USE_LLM" = "true" ] || [ "$USE_LLM" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --use_llm"
fi

# Gemini Configuration
if [ -n "$GEMINI_API_KEY" ] || [ -n "$GOOGLE_API_KEY" ]; then
    SERVER_CMD="$SERVER_CMD --gemini_api_key ${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
fi
if [ -n "$GEMINI_MODEL_NAME" ]; then
    SERVER_CMD="$SERVER_CMD --gemini_model_name $GEMINI_MODEL_NAME"
fi
if [ -n "$THINKING_BUDGET" ]; then
    SERVER_CMD="$SERVER_CMD --thinking_budget $THINKING_BUDGET"
fi

# Google Vertex AI Configuration
if [ -n "$VERTEX_PROJECT_ID" ]; then
    SERVER_CMD="$SERVER_CMD --vertex_project_id $VERTEX_PROJECT_ID"
fi
if [ -n "$VERTEX_LOCATION" ]; then
    SERVER_CMD="$SERVER_CMD --vertex_location $VERTEX_LOCATION"
fi
if [ "$VERTEX_DEDICATED" = "true" ] || [ "$VERTEX_DEDICATED" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --vertex_dedicated"
fi

# Claude Configuration
if [ -n "$CLAUDE_API_KEY" ] || [ -n "$ANTHROPIC_API_KEY" ]; then
    SERVER_CMD="$SERVER_CMD --claude_api_key ${CLAUDE_API_KEY:-$ANTHROPIC_API_KEY}"
fi
if [ -n "$CLAUDE_MODEL_NAME" ]; then
    SERVER_CMD="$SERVER_CMD --claude_model_name $CLAUDE_MODEL_NAME"
fi
if [ -n "$MAX_CLAUDE_TOKENS" ]; then
    SERVER_CMD="$SERVER_CMD --max_claude_tokens $MAX_CLAUDE_TOKENS"
fi

# OpenAI Configuration
if [ -n "$OPENAI_API_KEY" ]; then
    SERVER_CMD="$SERVER_CMD --openai_api_key $OPENAI_API_KEY"
fi
if [ -n "$OPENAI_MODEL" ]; then
    SERVER_CMD="$SERVER_CMD --openai_model $OPENAI_MODEL"
fi
if [ -n "$OPENAI_BASE_URL" ]; then
    SERVER_CMD="$SERVER_CMD --openai_base_url $OPENAI_BASE_URL"
fi
if [ -n "$OPENAI_IMAGE_FORMAT" ]; then
    SERVER_CMD="$SERVER_CMD --openai_image_format $OPENAI_IMAGE_FORMAT"
fi

# Azure OpenAI Configuration
if [ -n "$AZURE_ENDPOINT" ]; then
    SERVER_CMD="$SERVER_CMD --azure_endpoint $AZURE_ENDPOINT"
fi
if [ -n "$AZURE_API_KEY" ]; then
    SERVER_CMD="$SERVER_CMD --azure_api_key $AZURE_API_KEY"
fi
if [ -n "$AZURE_API_VERSION" ]; then
    SERVER_CMD="$SERVER_CMD --azure_api_version $AZURE_API_VERSION"
fi
if [ -n "$DEPLOYMENT_NAME" ]; then
    SERVER_CMD="$SERVER_CMD --deployment_name $DEPLOYMENT_NAME"
fi

# Ollama Configuration
if [ -n "$OLLAMA_BASE_URL" ]; then
    SERVER_CMD="$SERVER_CMD --ollama_base_url $OLLAMA_BASE_URL"
fi
if [ -n "$OLLAMA_MODEL" ]; then
    SERVER_CMD="$SERVER_CMD --ollama_model $OLLAMA_MODEL"
fi

# LLM Request Configuration
if [ -n "$TIMEOUT" ]; then
    SERVER_CMD="$SERVER_CMD --timeout $TIMEOUT"
fi
if [ -n "$MAX_RETRIES" ]; then
    SERVER_CMD="$SERVER_CMD --max_retries $MAX_RETRIES"
fi
if [ -n "$RETRY_WAIT_TIME" ]; then
    SERVER_CMD="$SERVER_CMD --retry_wait_time $RETRY_WAIT_TIME"
fi
if [ -n "$MAX_OUTPUT_TOKENS" ]; then
    SERVER_CMD="$SERVER_CMD --max_output_tokens $MAX_OUTPUT_TOKENS"
fi

# Processing Options (OpenWebUI compatible)
if [ "$SKIP_CACHE" = "true" ] || [ "$SKIP_CACHE" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --skip_cache"
fi
if [ "$FORCE_OCR" = "true" ] || [ "$FORCE_OCR" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --force_ocr"
fi
if [ "$PAGINATE" = "true" ] || [ "$PAGINATE" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --paginate"
fi
if [ "$DISABLE_IMAGE_EXTRACTION" = "true" ] || [ "$DISABLE_IMAGE_EXTRACTION" = "1" ]; then
    SERVER_CMD="$SERVER_CMD --disable_image_extraction"
fi

# Output format
if [ -n "$OUTPUT_FORMAT" ]; then
    SERVER_CMD="$SERVER_CMD --output_format $OUTPUT_FORMAT"
fi

# Additional configuration from environment
if [ -n "$LANGS" ]; then
    SERVER_CMD="$SERVER_CMD --langs $LANGS"
fi
if [ -n "$BATCH_MULTIPLIER" ]; then
    SERVER_CMD="$SERVER_CMD --batch_multiplier $BATCH_MULTIPLIER"
fi
if [ -n "$MAX_PAGES" ]; then
    SERVER_CMD="$SERVER_CMD --max_pages $MAX_PAGES"
fi

echo ""
echo "Starting Marker PDF Server..."
echo "Device: ${TORCH_DEVICE}"
echo "Port: ${PORT:-8001}"
echo "LLM Enabled: ${USE_LLM:-false}"
if [ -n "$LLM_SERVICE" ]; then
    echo "LLM Service: $LLM_SERVICE"
fi
echo ""
echo "Server will be available at: http://localhost:${PORT:-8001}"
echo "API Documentation: http://localhost:${PORT:-8001}/docs"
echo ""
echo "======================================"

# Execute the command
exec $SERVER_CMD
