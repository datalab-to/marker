import base64
import io
import re
from collections import Counter
from typing import Annotated, Optional, Tuple, Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel

from marker.schema import BlockTypes
from marker.schema.blocks.base import BlockId, BlockOutput
from marker.schema.document import Document
from marker.settings import settings
from marker.util import assign_config

# Import OSS uploader
from marker.oss_uploader import S3Client
s3_client = S3Client()
S3_AVAILABLE = True


class BaseRenderer:
    image_blocks: Annotated[
        Tuple[BlockTypes, ...], 
        "The block types to consider as images."
    ] = (BlockTypes.Picture, BlockTypes.Figure, BlockTypes.Molecule, BlockTypes.MoleculeTable)
    extract_images: Annotated[bool, "Extract images from the document."] = True
    image_extraction_mode: Annotated[
        Literal["lowres", "highres"],
        "The mode to use for extracting images.",
    ] = "highres"


    def __init__(self, config: Optional[BaseModel | dict] = None):
        assign_config(self, config)

    def __call__(self, document):
        # Children are in reading order
        raise NotImplementedError

    def extract_image(self, document: Document, image_id, to_base64=False):
        print(f"🖼️  [DEBUG] BaseRenderer.extract_image() called for {image_id}")
        
        image_block = document.get_block(image_id)
        if image_block is None:
            print(f"❌ [DEBUG] Image block not found for {image_id}")
            return None
            
        print(f"✅ [DEBUG] Found image block: {type(image_block).__name__} (type: {image_block.block_type})")
        
        cropped = image_block.get_image(document, highres=self.image_extraction_mode == "highres")
        
        if cropped is None:
            print(f"❌ [DEBUG] Failed to get image from block {image_id}")
            return None
            
        print(f"✅ [DEBUG] Got cropped image: {cropped.size}")

        if to_base64:
            image_buffer = io.BytesIO()
            cropped.save(image_buffer, format=settings.OUTPUT_IMAGE_FORMAT)
            cropped = base64.b64encode(image_buffer.getvalue()).decode(settings.OUTPUT_ENCODING)
            print(f"✅ [DEBUG] Converted to base64 (length: {len(cropped)})")
        
        return cropped

    def upload_image_to_s3(self, image, image_name, image_type="image", metadata=None):
        """
        Upload image to S3 if available, otherwise return None
        
        Args:
            image: PIL Image object
            image_name: Original image name
            image_type: Type of image (image, molecule, etc.)
            metadata: Additional metadata (currently not used in S3 implementation)
            
        Returns:
            S3 upload result dict with 'url' and 'key', or None
        """
        print(f"☁️  [DEBUG] upload_image_to_s3() called: {image_name} (type: {image_type})")
        
        if image is None:
            print(f"❌ [DEBUG] Image is None, cannot upload")
            return None
            
        print(f"🖼️  [DEBUG] Image size: {image.size}")
        
        if S3_AVAILABLE and s3_client:
            try:
                print(f"✅ [DEBUG] S3 client available, starting upload...")
                
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=85, optimize=True)
                image_data = img_buffer.getvalue()
                
                print(f"📦 [DEBUG] Image converted to bytes: {len(image_data)} bytes")
                
                # Upload to S3
                result = s3_client.s3_upload_from_file(image_name, image_data)
                
                if result and 'url' in result:
                    print(f"✅ [DEBUG] Image uploaded successfully: {result['url']}")
                    return result
                else:
                    print(f"❌ [DEBUG] Failed to upload image: {image_name}, result: {result}")
                    return None
                    
            except Exception as e:
                print(f"❌ [DEBUG] Failed to upload image to S3: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"❌ [DEBUG] S3 not available (S3_AVAILABLE: {S3_AVAILABLE}, s3_client: {s3_client is not None})")
        
        return None

    @staticmethod
    def merge_consecutive_math(html, tag="math"):
        if not html:
            return html
        pattern = fr'-</{tag}>(\s*)<{tag}>'
        html = re.sub(pattern, " ", html)

        pattern = fr'-</{tag}>(\s*)<{tag} display="inline">'
        html = re.sub(pattern, " ", html)
        return html

    @staticmethod
    def merge_consecutive_tags(html, tag):
        pattern = f'</{tag}>(\s*)<{tag}>'
        html = re.sub(pattern, r'\1', html)
        return html

    def get_page_footer(self, page: any):
        try:
            for block in page.children:
                if block.block_type == BlockTypes.PageFooter:
                    return block.raw_text(page)
        except Exception as e:
            print('get_page_footer', e, flush=True)
        return ''
    
    def get_page_header(self, page: any):
        try:
            for block in page.children:
                if block.block_type == BlockTypes.PageHeader:
                    return block.raw_text(page)
        except Exception as e:
            print('get_page_header', e, flush=True)
        return ''

    def generate_page_stats(self, document: Document, document_output):
        page_stats = []
        for page in document.pages:
            block_counts = Counter([str(block.block_type) for block in page.children]).most_common()
            block_metadata = page.aggregate_block_metadata()
            page_header = self.get_page_header(page)
            page_footer = self.get_page_footer(page)
            page_stats.append({
                "page_id": page.page_id,
                "text_extraction_method": page.text_extraction_method,
                "block_counts": block_counts,
                "block_metadata": block_metadata.model_dump(),
                "page_header": page_header,
                "page_footer": page_footer
            })
        return page_stats

    def generate_document_metadata(self, document: Document, document_output):
        metadata = {
            "table_of_contents": document.table_of_contents,
            "page_stats": self.generate_page_stats(document, document_output),
        }
        if document.debug_data_path is not None:
            metadata["debug_data_path"] = document.debug_data_path

        return metadata

    def extract_block_html(self, document: Document, block_output: BlockOutput):
        soup = BeautifulSoup(block_output.html, 'html.parser')

        content_refs = soup.find_all('content-ref')
        ref_block_id = None
        images = {}
        for ref in content_refs:
            src = ref.get('src')
            sub_images = {}
            for item in block_output.children:
                if item.id == src:
                    content, sub_images_ = self.extract_block_html(document, item)
                    sub_images.update(sub_images_)
                    ref_block_id: BlockId = item.id
                    break

            if ref_block_id.block_type in self.image_blocks and self.extract_images:
                image = self.extract_image(document, ref_block_id, to_base64=False)
                image_name = f"{ref_block_id.to_path()}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
                
                # Try to upload to S3 first
                image_block = document.get_block(ref_block_id)
                metadata = {}
                
                # Check if it's a molecule image
                if ref_block_id.block_type == BlockTypes.Molecule:
                    if hasattr(image_block, 'structure_data') and image_block.structure_data:
                        metadata = image_block.structure_data
                    s3_result = self.upload_image_to_s3(image, image_name, "molecule", metadata)
                else:
                    s3_result = self.upload_image_to_s3(image, image_name, "image", metadata)
                
                if s3_result:
                    # Store S3 URL information
                    images[ref_block_id] = {"url": s3_result['url'], "type": "s3", "original_name": image_name}
                else:
                    # Fall back to base64
                    images[ref_block_id] = self.extract_image(document, ref_block_id, to_base64=True)
            else:
                images.update(sub_images)
                ref.replace_with(BeautifulSoup(content, 'html.parser'))

        if block_output.id.block_type in self.image_blocks and self.extract_images:
            image = self.extract_image(document, block_output.id, to_base64=False)
            image_name = f"{block_output.id.to_path()}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
            
            # Try to upload to S3 first
            image_block = document.get_block(block_output.id)
            metadata = {}
            
            # Check if it's a molecule image
            if block_output.id.block_type == BlockTypes.Molecule:
                if hasattr(image_block, 'structure_data') and image_block.structure_data:
                    metadata = image_block.structure_data
                s3_result = self.upload_image_to_s3(image, image_name, "molecule", metadata)
            else:
                s3_result = self.upload_image_to_s3(image, image_name, "image", metadata)
            
            if s3_result:
                # Store S3 URL information
                images[block_output.id] = {"url": s3_result['url'], "type": "s3", "original_name": image_name}
            else:
                # Fall back to base64
                images[block_output.id] = self.extract_image(document, block_output.id, to_base64=True)

        return str(soup), images
