import os
import sys
import argparse
import glob
import time

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output
from marker.config.parser import ConfigParser


def convert_single_pdf(pdf_path: str, output_dir: str, config_options: dict = None):
    """
    Convert a single PDF file to markdown
    """
    if config_options is None:
        config_options = {}
    
    config_options["output_dir"] = output_dir
    
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    config_parser = ConfigParser(config_options)
    
    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    
    rendered = converter(pdf_path)
    out_folder = config_parser.get_output_folder(pdf_path)
    base_name = config_parser.get_base_filename(pdf_path)
    
    save_output(rendered, out_folder, base_name)
    
    return out_folder


def convert_directory(input_dir: str, output_dir: str, config_options: dict = None):
    """
    Convert all PDF files in a directory to markdown
    """
    input_dir = os.path.abspath(input_dir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(input_dir, "*.PDF")))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s) in {input_dir}")
    print(f"Output will be saved to {output_dir}")
    print("=" * 60)
    
    converted = []
    total_start = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n[{i}/{len(pdf_files)}] Converting: {pdf_name}")
        
        try:
            start_time = time.time()
            out_folder = convert_single_pdf(pdf_path, output_dir, config_options)
            elapsed = time.time() - start_time
            print(f"    Saved to: {out_folder}")
            print(f"    Time taken: {elapsed:.2f} seconds")
            converted.append((pdf_path, out_folder))
        except Exception as e:
            print(f"    Error converting {pdf_name}: {e}")
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"Total files converted: {len(converted)}/{len(pdf_files)}")
    print(f"Total time taken: {total_elapsed:.2f} seconds")
    
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using marker-pdf"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing PDF files to convert"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Directory to save output files (default: ./output)"
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "html", "chunks"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    parser.add_argument(
        "--disable-image-extraction",
        action="store_true",
        help="Disable image extraction from PDF"
    )
    parser.add_argument(
        "--page-range",
        type=str,
        default=None,
        help="Page range to convert (e.g., '0,5-10,20')"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR processing on all pages"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if args.output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_options = {
        "output_format": args.output_format,
        "disable_image_extraction": args.disable_image_extraction,
        "debug": args.debug,
    }
    
    if args.page_range:
        config_options["page_range"] = args.page_range
    
    if args.force_ocr:
        config_options["force_ocr"] = args.force_ocr
    
    print("=" * 60)
    print("PDF to Markdown Converter")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.output_format}")
    print()
    
    convert_directory(input_dir, output_dir, config_options)


if __name__ == "__main__":
    main()
