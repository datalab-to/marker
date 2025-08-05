from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Molecule(Block):
    block_type: BlockTypes = BlockTypes.Molecule
    block_description: str = "A chemical molecule structure or formula."
    html: str | None = None
    replace_output_newlines: bool = True
    structure_data: dict = {}
    confidence: float = 1.0
    
    def get_image(self, document, highres=True):
        """
        Extract image of the molecule from the document
        
        Args:
            document: Document object
            highres: Whether to use high resolution image
            
        Returns:
            PIL Image of the molecule
        """
        # Get the page containing this molecule
        page = document.get_page(self.page_id)
        if page is None:
            print(f"❌ [DEBUG] Page {self.page_id} not found in document")
            return None

        # Get the page image
        page_image = page.get_image()
        if page_image is None:
            print(f"❌ [DEBUG] Failed to get page image for page {self.page_id}")
            return None

        # Crop the molecule region from the page image
        bbox = self.polygon.bbox
        print('self.polygon.bbox', self.polygon, bbox, flush=True)
        print('image', page_image.size, flush=True)

        if len(bbox) >= 4:
            # bbox format: [x1, y1, x2, y2] - these are relative coordinates (0-1)
            x1, y1, x2, y2 = bbox
            ori_width = page.polygon.width
            ori_height = page.polygon.height
            x1 = x1 / ori_width * page_image.width
            y1 = y1 / ori_height * page_image.height
            x2 = x2 / ori_width * page_image.width
            y2 = y2 / ori_height * page_image.height
            
            # Check if crop area is valid
            if x2 <= x1 or y2 <= y1:
                print(f"❌ [DEBUG] Invalid crop area: width={x2-x1}, height={y2-y1}")
                return None
            
            # Crop the image
            cropped = page_image.crop((x1, y1, x2, y2))

            return cropped
        else:
            print(f"❌ [DEBUG] Invalid bbox length: {len(bbox)}")
            
        return None
    
    def assemble_html(self, document, child_blocks, parent_structure):
        # Use consistent placeholder ID format matching HTMLRenderer
        imgid = str(self.id)
        
        # 如果有自定义html
        if self.html:
            return f"<p>{self.html}</p>"
        
        # 如果有结构数据中的内容
        if self.structure_data.get('content'):
            return f"<p>{self.structure_data['content']}</p>"
        
        # 如果有SMILES数据
        if self.structure_data.get('smiles'):
            smiles = self.structure_data['smiles']
            return f"<p>_placeholder_molid_{imgid}_label_{self.structure_data.get('label', '')}_smiles_{smiles}</p>"
        
        # 默认情况使用placeholder
        return f"<p>_placeholder_molid_{imgid}</p>"


class MoleculeTable(Block):
    block_type: BlockTypes = BlockTypes.MoleculeTable
    block_description: str = "A table containing chemical molecules or molecular data."
    html: str | None = None
    replace_output_newlines: bool = True
    structure_data: dict = {}
    table_data: dict = {}
    confidence: float = 1.0
    
    def get_image(self, document, highres=True):
        """
        Extract image of the molecule table from the document
        
        Args:
            document: Document object
            highres: Whether to use high resolution image
            
        Returns:
            PIL Image of the molecule table
        """
        # Get the page containing this table
        page = document.get_page(self.page_id)
        if page is None:
            print(f"❌ [DEBUG] Page {self.page_id} not found in document")
            return None
            
        # Get the page image
        page_image = page.get_image(highres=highres)
        if page_image is None:
            print(f"❌ [DEBUG] Failed to get page image for page {self.page_id}")
            return None

        # Crop the table region from the page image
        bbox = self.polygon.bbox

        if len(bbox) >= 4:
            # bbox format: [x1, y1, x2, y2] - these are relative coordinates (0-1)
            x1_rel, y1_rel, x2_rel, y2_rel = bbox
            
            # Convert relative coordinates to absolute coordinates
            page_width, page_height = page_image.size
            x1 = x1_rel * page_width
            y1 = y1_rel * page_height
            x2 = x2_rel * page_width
            y2 = y2_rel * page_height
            
            # Ensure coordinates are within image bounds and are integers
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(page_width, int(x2))
            y2 = min(page_height, int(y2))
            
            # Check if crop area is valid
            if x2 <= x1 or y2 <= y1:
                print(f"❌ [DEBUG] Invalid crop area: width={x2-x1}, height={y2-y1}")
                return None
            
            # Crop the image
            cropped = page_image.crop((x1, y1, x2, y2))
            return cropped
        else:
            print(f"❌ [DEBUG] Invalid bbox length: {len(bbox)}")
            
        return None
    
    def assemble_html(self, document, child_blocks, parent_structure):
        # Use consistent placeholder ID format matching HTMLRenderer
        imgid = str(self.id)
        
        # 如果是mock数据，输出固定内容
        if self.table_data.get('mock', False):
            return f"<p>_placeholder_tableid_{imgid}</p>"
        
        # 如果有自定义html
        if self.html:
            # return f"{self.html}"
            return f"<p>_placeholder_tableid_{imgid}</p>"
        
        # 如果有表格数据中的内容
        if self.table_data.get('content'):
            # return f"<table>{self.table_data['content']}</table>"
            return f"<p>_placeholder_tableid_{imgid}</p>"
        
        # Use placeholder for default case
        return f"<p>_placeholder_tableid_{imgid}</p>" 