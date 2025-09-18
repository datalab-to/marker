import os
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import save_output
from marker.converters.pdf import PdfConverter
from marker.services.llama_cpp import LlamaCPPService

# Create model dictionary
models = create_model_dict()

# Configuration
config_dict = {
    "output_format": "markdown",
    "extract_images": True,
}

# Create converter with LLM service
converter = PdfConverter(
    config=config_dict,
    artifact_dict=models,
    llm_service="marker.services.llama_cpp.LlamaCPPService"
)

# Convert file
fpath = "/mnt/data/projects/marker-xpu/testfiles/Letter of medical necessity.pdf"
rendered = converter(fpath)

# Save output
out_folder = "./output"
os.makedirs(out_folder, exist_ok=True)
save_output(rendered, out_folder, "Letter of medical necessity")
