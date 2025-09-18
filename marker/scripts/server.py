import traceback
import uuid
import click
import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered


import base64
from contextlib import asynccontextmanager
from typing import Optional, Annotated, Dict, Any
import asyncio
import io
from fastapi import FastAPI, Form, File, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

# In-memory storage for tracking conversion requests
request_storage: Dict[str, Any] = {}
app_data = {}


UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict(config={})

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20"
        ),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(
            description="Force OCR on all pages of the PDF.  Defaults to False.  This can lead to worse results if you have good text in your PDFs (which is true in most cases)."
        ),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description="Whether to paginate the output.  Defaults to False.  If set to True, each page of the output will be separated by a horizontal rule that contains the page number (2 newlines, {PAGE_NUMBER}, 48 - characters, 2 newlines)."
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(
            description="The format to output the text in.  Can be 'markdown', 'json', or 'html'.  Defaults to 'markdown'."
        ),
    ] = "markdown"
    use_llm: Annotated[
        bool,
        Field(
            description="Enable LLM processing for enhanced accuracy. Defaults to False."
        ),
    ] = False
    strip_existing_ocr: Annotated[
        bool,
        Field(
            description="Remove all existing OCR text in the document and re-OCR with surya. Defaults to False."
        ),
    ] = False
    disable_image_extraction: Annotated[
        bool,
        Field(
            description="Disable image extraction. Defaults to False."
        ),
    ] = False
    skip_cache: Annotated[
        bool,
        Field(
            description="Skip cache when processing. Defaults to False."
        ),
    ] = False
    format_lines: Annotated[
        bool,
        Field(
            description="Format lines in the output. Defaults to False."
        ),
    ] = False
    max_pages: Annotated[
        Optional[int],
        Field(
            description="Maximum number of pages to process. Defaults to None."
        ),
    ] = None


async def _convert_pdf(params: CommonParams):
    # Validate filepath
    if not params.filepath:
        return {
            "success": False,
            "error": "No file provided",
        }
    assert params.output_format in ["markdown", "json", "html", "chunks"], (
        "Invalid output format"
    )
    try:
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }

    encoded = {}
    for k, v in images.items():
        byte_stream = io.BytesIO()
        v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
        encoded[k] = base64.b64encode(byte_stream.getvalue()).decode(
            settings.OUTPUT_ENCODING
        )

    return {
        "format": params.output_format,
        "output": text,
        "images": encoded,
        "metadata": metadata,
        "success": True,
    }


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    return await _convert_pdf(params)


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename or "uploaded_file.pdf")
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        force_ocr=force_ocr or False,
        paginate_output=paginate_output or False,
        output_format=output_format or "markdown",
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results


async def process_conversion(request_id: str):
    """Process the PDF conversion in background and update the request storage with results."""
    try:
        # Get the request info
        request_info = request_storage[request_id]
        params = request_info["params"]
        
        # Perform the conversion
        result = await _convert_pdf(params)
        
        # Format the result to match Datalab API
        formatted_result = {
            "output_format": params.output_format,
            params.output_format: result["output"],
            "status": "complete",
            "success": result["success"],
            "images": result["images"],
            "metadata": result["metadata"] or {},
            "error": "" if result["success"] else result["error"],
            "page_count": len(result["metadata"].get("pages", [])) if result["metadata"] else 0,
        }
        
        # Update the request storage with the result
        request_storage[request_id]["status"] = "completed"
        request_storage[request_id]["result"] = formatted_result
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Update the request storage with the error
        request_storage[request_id]["status"] = "failed"
        request_storage[request_id]["error"] = str(e)


@app.post("/api/v1/marker")
async def convert_pdf_datalab_format(
    paginate: Optional[str] = Form(default="false"),
    output_format: Optional[str] = Form(default="markdown"),
    force_ocr: Optional[str] = Form(default="false"),
    use_llm: Optional[str] = Form(default="false"),
    strip_existing_ocr: Optional[str] = Form(default="false"),
    disable_image_extraction: Optional[str] = Form(default="false"),
    skip_cache: Optional[str] = Form(default="false"),
    format_lines: Optional[str] = Form(default="false"),
    max_pages: Optional[int] = Form(default=None),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    # Generate a unique request ID
    request_id = str(uuid.uuid4())
    
    # Save the uploaded file
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename or "uploaded_file.pdf")
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)
    
    # Map Datalab parameters to internal parameters
    # Convert string parameters to boolean values
    def str_to_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")
    
    params = CommonParams(
        filepath=upload_path,
        page_range=None,  # Not directly supported in Datalab API
        force_ocr=str_to_bool(force_ocr) if force_ocr else False,
        paginate_output=str_to_bool(paginate) if paginate else False,
        output_format=output_format or "markdown",
        use_llm=str_to_bool(use_llm) if use_llm else False,
        strip_existing_ocr=str_to_bool(strip_existing_ocr) if strip_existing_ocr else False,
        disable_image_extraction=str_to_bool(disable_image_extraction) if disable_image_extraction else False,
        skip_cache=str_to_bool(skip_cache) if skip_cache else False,
        format_lines=str_to_bool(format_lines) if format_lines else False,
        max_pages=max_pages,
    )
    
    # Store request info for polling
    request_storage[request_id] = {
        "status": "processing",
        "file_path": upload_path,
        "params": params,
    }
    
    # Process the conversion in background
    asyncio.create_task(process_conversion(request_id))
    
    # Return initial response
    request_check_url = f"http://localhost:8000/api/v1/marker/{request_id}"
    return {
        "success": True,
        "error": None,
        "request_id": request_id,
        "request_check_url": request_check_url,
    }


@app.get("/api/v1/marker/{request_id}")
async def get_conversion_result(request_id: str):
    # Check if request exists
    if request_id not in request_storage:
        return {
            "success": False,
            "error": "Request not found",
        }
    
    request_info = request_storage[request_id]
    
    # If still processing, return processing status
    if request_info["status"] == "processing":
        return {
            "status": "processing",
            "success": True,
            "error": "",
        }
    
    # If completed, return result
    if request_info["status"] == "completed":
        result = request_info["result"]
        # Clean up the stored file
        if os.path.exists(request_info["file_path"]):
            os.remove(request_info["file_path"])
        # Remove from storage
        del request_storage[request_id]
        return result
    
    # If failed, return error
    if request_info["status"] == "failed":
        # Clean up the stored file
        if os.path.exists(request_info["file_path"]):
            os.remove(request_info["file_path"])
        # Remove from storage
        del request_storage[request_id]
        return {
            "status": "failed",
            "success": False,
            "error": request_info.get("error", "Conversion failed"),
        }
    
    return {
        "success": False,
        "error": "Invalid request status",
    }


@click.command()
@click.option("--port", type=int, default=8015, help="Port to run the server on")
@click.option("--host", type=str, default="0.0.0.0", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn

    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
    )
