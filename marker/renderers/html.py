import textwrap

from PIL import Image
from typing import Annotated, Literal, Tuple

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from pydantic import BaseModel

from marker.renderers import BaseRenderer
from marker.schema import BlockTypes
from marker.schema.blocks import BlockId
from marker.settings import settings

# Ignore beautifulsoup warnings
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Suppress DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

# Import OSS uploader
from marker.oss_uploader import S3Client
s3_client = S3Client()
S3_AVAILABLE = True


class HTMLOutput(BaseModel):
    html: str
    images: dict
    metadata: dict


class HTMLRenderer(BaseRenderer):
    """
    A renderer for HTML output.
    """
    page_blocks: Annotated[
        Tuple[BlockTypes],
        "The block types to consider as pages.",
    ] = (BlockTypes.Page,)
    paginate_output: Annotated[
        bool,
        "Whether to paginate the output.",
    ] = False

    def extract_image(self, document, image_id):
        image_block = document.get_block(image_id)
        cropped = image_block.get_image(document, highres=self.image_extraction_mode == "highres")
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
        if S3_AVAILABLE and s3_client:
            try:
                # Convert PIL Image to bytes
                import io
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=85, optimize=True)
                image_data = img_buffer.getvalue()
                
                # Upload to S3
                result = s3_client.s3_upload_from_file(image_name, image_data)
                
                if result and 'url' in result:
                    print(f"✅ Image uploaded successfully: {result['url']}")
                    return result
                else:
                    print(f"❌ Failed to upload image: {image_name}")
                    return None
                    
            except Exception as e:
                print(f"Failed to upload image to S3: {e}")
                return None
        return None

    def extract_html(self, document, document_output, level=0):
        soup = BeautifulSoup(document_output.html, 'html.parser')
        content_refs = soup.find_all('content-ref')
        ref_block_id = None
        images = {}

        for ref in content_refs:
            src = ref.get('src')
            sub_images = {}
            content = ""
            for item in document_output.children:
                if item.id == src:
                    content, sub_images_ = self.extract_html(document, item, level + 1)
                    sub_images.update(sub_images_)
                    ref_block_id: BlockId = item.id
                    break

            if ref_block_id.block_type in self.image_blocks:
                if self.extract_images:
                    image = self.extract_image(document, ref_block_id)
                    image_name = f"{ref_block_id.to_path()}.{settings.OUTPUT_IMAGE_FORMAT.lower()}"
                    
                    # Try to upload to S3 first
                    image_block = document.get_block(ref_block_id)
                    metadata = {}
                    img_id = str(ref_block_id)
                    img_tag = ""  # Initialize img_tag
                    
                    # Check if it's a molecule image
                    if ref_block_id.block_type == BlockTypes.Molecule:
                        if hasattr(image_block, 'structure_data') and image_block.structure_data:
                            metadata = image_block.structure_data
                        s3_result = self.upload_image_to_s3(image, image_name, "molecule", metadata)
                        # print("s3_result: ", s3_result, img_id)
                        if s3_result:
                            # Use S3 URL and custom molecule tag
                            images[img_id] = {
                                "url": s3_result['url'], 
                                "type": "s3", 
                                "key": s3_result['key'], 
                                'extra_type': 'molecule_img',
                                "smiles": image_block.structure_data.get('smiles', ''),
                                "mol_block": image_block.structure_data.get('mol_block', ''),
                                "label": image_block.structure_data.get('label', ''),
                                "page_idx": image_block.structure_data.get('page_idx', ''),
                                "bbox": image_block.structure_data.get('bbox', [])
                            }
                            img_tag = f'<img src="{s3_result["url"]}" alt="Molecule {img_id}" data-type="molecule"/>'
                        else:
                            # Fall back to local molecule image
                            images[img_id] = image
                            img_tag = f'<img src="{image_name}" alt="Molecule {img_id}" data-type="molecule"/>'
                    
                    elif ref_block_id.block_type == BlockTypes.MoleculeTable:
                        # s3_result = self.upload_image_to_s3(image, image_name, "molecule_table", metadata)
                        
                        # if s3_result:
                            # Use S3 URL and custom molecule table tag
                        images[img_id] = {
                            "url": '', 
                            "type": "s3", 
                            "key": '', 
                            "extra_type": "molecule_table",
                            "html_content": image_block.html,
                            "page_idx": image_block.structure_data.get('page_idx', ''),
                            "bbox": image_block.structure_data.get('bbox', [])
                        }
                        img_tag = f'<img src="" alt="Molecule Table {img_id}" data-type="molecule-table"/>'

                    elif ref_block_id.block_type == BlockTypes.Picture:
                        # Handle pictures
                        s3_result = self.upload_image_to_s3(image, image_name, "picture", metadata)
                        
                        if s3_result:
                            # Use S3 URL and custom picture tag
                            images[img_id] = {"url": s3_result['url'], "type": "s3", "key": s3_result['key'], 'extra_type': 'picture'}
                            img_tag = f'<img src="{s3_result["url"]}" alt="Picture {img_id}" data-type="picture"/>'
                        else:
                            # Fall back to local picture image
                            images[img_id] = image
                            img_tag = f'<img src="{image_name}" alt="Picture {img_id}" data-type="picture"/>'

                    elif ref_block_id.block_type == BlockTypes.Figure:
                        # Handle figures
                        s3_result = self.upload_image_to_s3(image, image_name, "figure", metadata)
                        
                        if s3_result:
                            # Use S3 URL and custom figure tag
                            images[img_id] = {"url": s3_result['url'], "type": "s3", "key": s3_result['key'], 'extra_type': 'picture'}
                          
                            img_tag = f'<picture id="{img_id}" data-type="picture"/>'
                        else:
                            # Fall back to local figure image
                            images[img_id] = image
                            img_tag = f'<picture id="{img_id}" data-type="picture"/>'
                    else:
                        # Other image types
                        s3_result = self.upload_image_to_s3(image, image_name, "image", metadata)
                        
                        if s3_result:
                            # Use S3 URL and custom image tag
                            images[img_id] = {"url": s3_result['url'], "type": "s3", "key": s3_result['key']}
                 
                            img_tag = f'<img src="{s3_result["url"]}" alt="Image {img_id}"/>'
                        else:
                            # Fall back to standard markdown image
                            images[img_id] = image
                            img_tag = f'<img src="{image_name}" alt="Image {img_id}"/>'
                    
                    # Replace the content-ref with the content and image tag
                    replacement_html = f"<p>{content}</p>" if content else f"<p>{img_tag}</p>"
                    # replacement_html = f"{img_tag}"

                    ref.replace_with(BeautifulSoup(replacement_html, 'html.parser'))
                else:
                    # This will be the image description if using llm mode, or empty if not
                    ref.replace_with(BeautifulSoup(f"{content}", 'html.parser'))
            elif ref_block_id.block_type in self.page_blocks:
                images.update(sub_images)
                if self.paginate_output:
                    content = f"<div class='page' data-page-id='{ref_block_id.page_id}'>{content}</div>"
                ref.replace_with(BeautifulSoup(f"{content}", 'html.parser'))
            else:
                images.update(sub_images)
                ref.replace_with(BeautifulSoup(f"{content}", 'html.parser'))

        output = str(soup)
        if level == 0:
            output = self.merge_consecutive_tags(output, 'b')
            output = self.merge_consecutive_tags(output, 'i')
            output = self.merge_consecutive_math(output) # Merge consecutive inline math tags
            output = textwrap.dedent(f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="utf-8" />
                </head>
                <body>
                    {output}
                </body>
            </html>
""")
            print("@@@ images: ", images)

        return output, images

    def __call__(self, document) -> HTMLOutput:
        document_output = document.render()
        full_html, images = self.extract_html(document, document_output)
        soup = BeautifulSoup(full_html, 'html.parser')
        full_html = soup.prettify() # Add indentation to the HTML
        return HTMLOutput(
            html=full_html,
            images=images,
            metadata=self.generate_document_metadata(document, document_output)
        )
