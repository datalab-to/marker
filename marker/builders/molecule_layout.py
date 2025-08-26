from typing import Annotated, List, Dict, Any, Optional
import random

from marker.builders import BaseBuilder
from marker.providers.pdf import PdfProvider
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from marker.schema.blocks import Molecule, MoleculeTable
from marker.schema.polygon import PolygonBox
from marker.schema import BlockTypes

import copy
import sys
import traceback
from PIL import Image
from tqdm import tqdm
import os
import warnings
import io

# Try to import img2mol processor
IMG2MOL_AVAILABLE = True
# Suppress warnings
warnings.filterwarnings("ignore")


class MoleculeLayoutBuilder(BaseBuilder):
    """
    A builder for performing chemical molecule layout detection on PDF pages and merging the results into the document.
    Uses img2mol's Parser_Processer for molecule and table detection, or mock data for testing.
    采用单例模式来避免重复创建Parser_Processer实例，减少内存泄漏。
    """
    # 单例相关的类变量
    _instance = None
    _processor_cache = {}  # 用于缓存不同配置的processor实例
    
    # The overlap threshold for replacing existing blocks with molecule blocks
    overlap_threshold: float = 0.9
    
    # The overlap threshold for replacing table blocks with molecule table blocks  
    table_overlap_threshold: float = 0.9
    
    # Whether to disable the tqdm progress bar
    disable_tqdm: bool = False
    
    # Whether to use mock data instead of real img2mol detection
    use_mock_data: bool = False
    
    def __new__(cls, processor_config=None, config=None):
        """实现单例模式，确保相同配置只有一个实例"""
        # 创建配置的哈希键
        config_key = cls._create_config_key(processor_config, config)
        
        if config_key not in cls._processor_cache:
            instance = super(MoleculeLayoutBuilder, cls).__new__(cls)
            cls._processor_cache[config_key] = instance
            print(f"🆕 Created new MoleculeLayoutBuilder instance for config: {config_key}")
        else:
            print(f"♻️  Reusing existing MoleculeLayoutBuilder instance for config: {config_key}")
        
        return cls._processor_cache[config_key]
    
    @classmethod
    def _create_config_key(cls, processor_config, config):
        """创建配置的唯一键"""
        # 提取关键配置参数创建哈希
        import hashlib
        import json
        
        key_params = {}
        
        if processor_config:
            # 只选择影响模型加载的关键参数
            key_params.update({
                'device': processor_config.get('device', 'cuda'),
                'with_mol_detect': processor_config.get('with_mol_detect', True),
                'with_table_detect': processor_config.get('with_table_detect', True),
                'use_yolo_mol_model': processor_config.get('use_yolo_mol_model', True),
                'use_yolo_table_model': processor_config.get('use_yolo_table_model', True),
                'use_got_ocr_model': processor_config.get('use_got_ocr_model', True),
                'model_dir': processor_config.get('model_dir', 'default')
            })
        
        if config:
            key_params.update({
                'use_molecule_detection': config.get('use_molecule_detection', False)
            })
        
        # 创建哈希
        config_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @classmethod
    def clear_cache(cls):
        """清理所有缓存的实例"""
        print("🧹 Clearing MoleculeLayoutBuilder cache...")
        for config_key, instance in cls._processor_cache.items():
            if hasattr(instance, 'cleanup_memory'):
                instance.cleanup_memory()
        cls._processor_cache.clear()
        print("✅ MoleculeLayoutBuilder cache cleared")
    
    def __init__(self, processor_config=None, config=None):
        """
        初始化分子识别Layout Builder
        
        Args:
            processor_config: img2mol Parser_Processer的配置参数
            config: marker配置
        """
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            print("♻️  MoleculeLayoutBuilder already initialized, skipping...")
            return
        
        super().__init__(config)
        
        self.processor_config = processor_config or {}
        self.processor = None
        self._initialized = True
        
        # 检查是否使用mock模式
        self.use_mock_data = (
            self.processor_config.get('use_mock_data', False) or 
            not IMG2MOL_AVAILABLE or
            self.processor_config.get('mock_mode', False)
        )
        
        if not self.use_mock_data:
            self._initialize_processor()
        else:
            print("🎭 使用Mock模式进行分子检测测试")
    
    def _initialize_processor(self):
        """Initialize the img2mol Parser_Processer"""
        try:
            # Import img2mol processor
            import sys
            sys.path.append('/app/img2mol')
            sys.path.append('/app/img2mol/clean_img2smiles/src')
            from img2smiles.pipeline.Processer import Parser_Processer
            
            # Create processor with configuration
            self.processor = Parser_Processer(**self.processor_config)
            
            print("✅ Successfully initialized img2mol Parser_Processer")
            
        except Exception as e:
            traceback.print_exc()
            print(f"Warning: Failed to initialize img2mol processor: {e}")
            print("🎭 切换到Mock模式")
            self.use_mock_data = True
            self.processor = None
    
    def cleanup_memory(self):
        """清理内存"""
        try:
            if self.processor and hasattr(self.processor, 'cleanup_memory'):
                self.processor.cleanup_memory()
            
            self.processor = None
            
            # 从全局模型管理器导入并清理
            try:
                import sys
                sys.path.append('/app/img2mol')
                sys.path.append('/app/img2mol/clean_img2smiles/src')
                from img2smiles.pipeline.model_manager import model_manager
                model_manager.print_memory_stats()
            except:
                pass
            
            print("✅ MoleculeLayoutBuilder memory cleaned up")
            
        except Exception as e:
            print(f"⚠️ Warning during MoleculeLayoutBuilder cleanup: {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup_memory()
        except:
            pass

    def __call__(self, document: Document, provider: PdfProvider):
        """Process all pages in the document to detect molecules and tables"""
        if self.use_mock_data:
            detection_results = self.generate_mock_detection_results(document.pages)
        else:
            if self.processor is None:
                print("Molecule processor not available, skipping molecule detection")
                return
            detection_results = self.detect_molecules_and_tables(document.pages)
        
        self.merge_molecule_blocks_to_pages(document.pages, detection_results)

    def generate_mock_detection_results(self, pages: List[PageGroup]) -> List[dict]:
        """
        生成Mock检测结果用于测试
        随机生成一些分子结构，基于已有表格生成分子表格（坐标微调用于测试覆盖功能）
        """
        results = []
        
        for page_idx, page in enumerate(tqdm(pages, disable=self.disable_tqdm, desc="Mock分子检测")):
            molecules = []
            tables = []
            
            # 获取页面尺寸（用于生成合理的bbox）
            page_width = 800  # 默认值
            page_height = 1000  # 默认值
            
            if hasattr(page, 'page_image') and page.page_image:
                if hasattr(page.page_image, 'size'):
                    page_width, page_height = page.page_image.size
                elif hasattr(page.page_image, 'shape'):
                    page_height, page_width = page.page_image.shape[:2]
            
            # 随机生成2-4个分子结构
            num_molecules = random.randint(2, 4)
            for _ in range(num_molecules):
                # 生成随机bbox
                x1 = random.randint(50, page_width - 200)
                y1 = random.randint(50, page_height - 200)
                x2 = x1 + random.randint(80, 150)
                y2 = y1 + random.randint(80, 150)
                
                molecules.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': random.uniform(0.8, 0.95),
                    'data': {
                        'bbox': [x1, y1, x2, y2],
                        'smiles': 'c1ccccc1',  # 苯环的SMILES
                        'mock': True
                    }
                })
            
            # 只基于已有的Table blocks生成分子表格mock数据（坐标微调，内容替换）
            existing_tables = []
            if hasattr(page, 'children'):
                existing_tables = [b for b in page.children if hasattr(b, 'block_type') and b.block_type == BlockTypes.Table]
            elif hasattr(page, 'blocks'):
                existing_tables = [b for b in page.blocks if b.block_type == BlockTypes.Table]
            
            if existing_tables:
                print(f"📋 页面 {page_idx + 1}: 发现 {len(existing_tables)} 个已有表格，将生成对应的分子表格mock数据")
                
                for table_block in existing_tables:
                    # 获取原始表格的坐标
                    original_bbox = table_block.polygon.bbox
                    x1, y1, x2, y2 = original_bbox
                    
                    # 微调坐标（稍微偏移，确保有足够重叠来触发替换）
                    # 偏移范围：-5到+5像素，确保90%以上重叠
                    offset_x = random.randint(-5, 5)
                    offset_y = random.randint(-5, 5)
                    
                    adjusted_bbox = [
                        x1 + offset_x,
                        y1 + offset_y, 
                        x2 + offset_x,
                        y2 + offset_y
                    ]
                    
                    # 生成HTML格式的分子表格内容
                    mock_html_table = self._generate_mock_molecule_table_html()
                    
                    tables.append({
                        'bbox': adjusted_bbox,
                        'confidence': random.uniform(0.85, 0.95),
                        'data': {
                            'bbox': adjusted_bbox,
                            'original_bbox': original_bbox,  # 保存原始坐标用于调试
                            'table_type': 'molecule_table',
                            'html_content': mock_html_table,
                            'format': 'html',
                            'mock': True,
                            'source': 'existing_table_adjusted'  # 标记数据来源
                        }
                    })
            else:
                print(f"📋 页面 {page_idx + 1}: 未发现已有表格，跳过分子表格生成")
            
            results.append({
                'page_idx': page_idx,
                'molecules': molecules,
                'tables': tables
            })
            
            print(f"📄 页面 {page_idx + 1}: Mock生成 {len(molecules)} 个分子, {len(tables)} 个分子表格 (基于已有表格)")
        
        return results

    def _generate_mock_molecule_table_html(self):
        """
        生成Mock分子表格的HTML内容
        包含化学分子结构数据，cell里填入C1CCCCC1等SMILES
        """
        # 定义一些常见的分子SMILES
        molecules = [
            'C1CCCCC1',      # 环己烷
            'c1ccccc1',      # 苯
            'CCO',           # 乙醇  
            'CC(=O)O',       # 乙酸
            'CC(C)C',        # 异丙烷
            'C1=CC=CC=C1O',  # 苯酚
            'CCN',           # 乙胺
            'C1CCC(CC1)O'    # 环己醇
        ]
        
        # 随机选择表格大小（2-4行，2-3列）
        rows = random.randint(2, 4)
        cols = random.randint(2, 3)
        
        # 构建HTML表格
        html_parts = ['<table border="1" style="border-collapse: collapse;">']
        
        # 表头
        html_parts.append('<tr>')
        headers = ['化合物', 'SMILES', '分子量'] if cols == 3 else ['化合物', 'SMILES']
        for header in headers[:cols]:
            html_parts.append(f'<th style="padding: 8px; background-color: #f0f0f0;">{header}</th>')
        html_parts.append('</tr>')
        
        # 数据行
        for i in range(rows):
            html_parts.append('<tr>')
            mol_smiles = random.choice(molecules)
            
            for j in range(cols):
                if j == 0:  # 化合物名称列
                    content = f'化合物-{i+1}'
                elif j == 1:  # SMILES列
                    content = mol_smiles
                else:  # 分子量列
                    content = f'{random.randint(50, 300)}.{random.randint(10, 99)}'
                
                html_parts.append(f'<td style="padding: 8px; text-align: center;">{content}</td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        
        return ''.join(html_parts)

    def detect_molecules_and_tables(self, pages: List[PageGroup]) -> List[dict]:
        """
        Detect molecules and tables on each page using img2mol _prediction_from_pdf
        
        Returns:
            List of detection results for each page
        """
        results = []
        
        for page_idx, page in enumerate(tqdm(pages, disable=self.disable_tqdm, desc="Detecting molecules")):
            try:
                # Get page image
                page_image = page.get_image(highres=True)
                if page_image is None:
                    print(f"Warning: No image available for page {page_idx}")
                    results.append({'page_idx': page_idx, 'molecules': [], 'tables': []})
                    continue
                
                # Convert to PIL Image if needed
                if not isinstance(page_image, Image.Image):
                    if hasattr(page_image, 'image'):
                        page_image = page_image.image
                    else:
                        page_image = Image.fromarray(page_image)
                
                # Use _prediction_from_pdf method for comprehensive detection
                # This method handles both molecule and table detection in one call
                
                # Set up parameters for _prediction_from_pdf
                with_mol_detect = self.processor_config.get('with_mol_detect', True)
                with_table_detect = self.processor_config.get('with_table_detect', True)
                
                if with_mol_detect or with_table_detect:
                    # Call _prediction_from_pdf with image input
                    # Returns: total_result_dict, total_table_result_dict (if with_table=True)
                    # or just total_result_dict (if with_table=False)
                    prediction_result = self.processor._prediction_from_pdf(
                        image=page_image,
                        page_idx_list=[page_idx + 1],  # _prediction_from_pdf expects 1-based page indices
                        with_tta=True,
                        with_layout_parser=True,
                        use_coref=True,
                        use_ocr=True,
                        debug=False,
                        with_molscribe=True,
                        with_table=True,
                        with_ocr=True,
                        with_html=False,
                        with_expand_mol=False,
                        return_realative_coordinates=True,
                        quick_prediction=False,
                        mode='auto',
                        osd_detect=False,
                        return_table_html=False
                    )

                    # Parse the result based on return type
                    if with_table_detect:
                        # Returns (total_result_dict, total_table_result_dict)
                        if isinstance(prediction_result, tuple) and len(prediction_result) >= 2:
                            total_result_dict, total_table_result_dict = prediction_result[:2]
                        else:
                            total_result_dict = prediction_result
                            total_table_result_dict = {}
                    else:
                        # Returns just total_result_dict
                        total_result_dict = prediction_result
                        total_table_result_dict = {}
                    
                    # Process molecule results
                    molecules = []
                    page_key = page_idx + 1  # _prediction_from_pdf uses 1-based page indices
                    if page_key in total_result_dict:
                        for mol_result in total_result_dict[page_key]:
                            if 'mol_box' in mol_result:
                                mol_box = mol_result['mol_box']
                                # print('mol_result', mol_result, flush=True)
                                # Convert tuple to list format
                                bbox = [mol_box[0], mol_box[1], mol_box[2], mol_box[3]]
                                print('bbbbbbbbbbbox', bbox, flush=True)
                                
                                # Convert relative coordinates to absolute coordinates
                                page_width = page.polygon.width
                                page_height = page.polygon.height
                                absolute_bbox = [
                                    bbox[0] * page_width,   # x1 * width
                                    bbox[1] * page_height,  # y1 * height
                                    bbox[2] * page_width,   # x2 * width
                                    bbox[3] * page_height   # y2 * height
                                ]
                                print(f'absolute_bbox: {absolute_bbox}, page_size: {page_width}x{page_height}', flush=True)
                                
                                # Extract additional data from _prediction_from_pdf result
                                smiles = mol_result.get('post_SMILES', mol_result.get('Cano_SMILES', 'detected_molecule'))
                                
                                molecules.append({
                                    'bbox': absolute_bbox,
                                    'confidence': 0.9,  # Default confidence
                                    'data': {
                                        'page_idx': page_idx,
                                        'bbox': bbox,
                                        'label_box': mol_result.get('label_box_list', []),
                                        'label': '/'.join(mol_result.get('label_string', [])),
                                        'smiles': smiles,
                                        'mol_block': mol_result.get('post_molblock', ''),
                                        'assigned_idx': mol_result.get('assigned_idx', ''),
                                        'state': mol_result.get('state', 'unknown'),
                                        'mock': False
                                    }
                                })
                    
                    # Process table results
                    tables = []
                    if with_table_detect and page_key in total_table_result_dict:
                        for table_result in total_table_result_dict[page_key]:
                            if table_result:
                                # Extract HTML content if available
                                html_content = table_result.get('html', '<table><tr><td>Molecular Data Table</td></tr></table>')
                                # 如果没有smiles
                                if "Cano_SMILES" not in html_content:
                                    continue
                                print('table_result', table_result, table_result['box'], flush=True)
                                ori_bbox = table_result['box']
                                bbox = [ori_bbox[0], ori_bbox[1], ori_bbox[2], ori_bbox[3]]
                                print('ccccccccccbox', bbox, flush=True)
                                
                                # Compare with page dimensions to see if this is already absolute
                                page_width = page.polygon.width
                                page_height = page.polygon.height
                                absolute_bbox = [
                                    bbox[0] * page_width,   # x1 * width
                                    bbox[1] * page_height,  # y1 * height
                                    bbox[2] * page_width,   # x2 * width
                                    bbox[3] * page_height   # y2 * height
                                ]
                                print(f'table_bbox: {bbox}, page_size: {page_width}x{page_height}, bbox_range: x=[{bbox[0]}-{bbox[2]}], y=[{bbox[1]}-{bbox[3]}]', flush=True)
                                
                                tables.append({
                                    'bbox': absolute_bbox,
                                    'confidence': table_result.get('confidence', 0.9),
                                    'data': {
                                        'bbox': bbox,
                                        'page_idx': page_idx,
                                        'table_type': 'molecule_table',
                                        'html_content': html_content,
                                        'dataframe': table_result.get('dataframe', None),
                                        'has_Rgroup': table_result.get('has_Rgroup', False),
                                        'format': 'html',
                                        'mock': False,
                                        'source': 'prediction_from_pdf'
                                    }
                                })
                
                else:
                    molecules = []
                    tables = []
                
                results.append({
                    'page_idx': page_idx,
                    'molecules': molecules,
                    'tables': tables
                })
                
                print(f"📄 页面 {page_idx + 1}: 检测到 {len(molecules)} 个分子, {len(tables)} 个分子表格")
                
            except Exception as e:
                traceback.print_exc()
                print(f"Error detecting molecules/tables on page {page_idx}: {e}")
                results.append({'page_idx': page_idx, 'molecules': [], 'tables': []})
        
        return results

    def _bbox_to_polygon(self, bbox):
        """
        将bbox转换为polygon格式
        bbox格式: [x1, y1, x2, y2]
        polygon格式: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        """

        return PolygonBox.from_bbox(bbox)

    def merge_molecule_blocks_to_pages(self, pages: List[PageGroup], detection_results: List[dict]):
        """
        Merge detected molecules and tables into page structures
        
        Args:
            pages: List of page groups to modify
            detection_results: Detection results from img2mol or mock data
        """
        for page_result in detection_results:
            page_idx = page_result.get('page_idx', 0)
            if page_idx >= len(pages):
                continue
                
            page = pages[page_idx]
            new_blocks = []
            
            # Process molecule detections
            for molecule_detection in page_result.get('molecules', []):
                bbox = molecule_detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                    
                polygon = self._bbox_to_polygon(bbox)
                
                if self.use_mock_data:
                    # Mock数据
                    structure_data = {
                        'smiles': 'c1ccccc1',
                        'formula': 'C6H6',
                        'mock': True
                    }
                else:
                    # 真实数据
                    structure_data = molecule_detection.get('data', {})
                
                # Create molecule block with proper page_id
                mol_block = Molecule(
                    polygon=polygon,
                    page_id=page.page_id,
                    structure_data=structure_data,
                    confidence=molecule_detection.get('confidence', 1.0)
                )
                new_blocks.append(mol_block)
            
            # Process table detections  
            for table_detection in page_result.get('tables', []):
                bbox = table_detection.get('bbox', [])
                if len(bbox) != 4:
                    continue
                    
                polygon = self._bbox_to_polygon(bbox)
                table_data = table_detection.get('data', {})
                
                # 获取HTML内容
                html_content = table_data.get('html_content', '')
                
                # 调试信息
                source = table_data.get('source', 'unknown')
                original_bbox = table_data.get('original_bbox')
                if original_bbox:
                    print(f"🔄 基于已有表格生成分子表格: 原始坐标 {original_bbox} -> 调整后坐标 {bbox}")
                
                # Create molecule table block with proper page_id
                mol_table_block = MoleculeTable(
                    polygon=polygon,
                    page_id=page.page_id,
                    structure_data={'page_idx': page_idx, 'bbox': bbox, 'html_content': html_content},
                    html=html_content,  # 直接使用html字段
                    confidence=table_detection.get('confidence', 1.0)
                )
                new_blocks.append(mol_table_block)
            
            if new_blocks:
                # Replace overlapping blocks for molecules (any block type with high overlap)
                molecule_blocks = [b for b in new_blocks if isinstance(b, Molecule)]
                if molecule_blocks:
                    self._replace_overlapping_blocks(
                        page, 
                        molecule_blocks, 
                        self.overlap_threshold,
                        target_types=[BlockTypes.Figure, BlockTypes.Picture]
                    )
                
                # Replace overlapping blocks for tables (specifically target Table blocks) 
                table_blocks = [b for b in new_blocks if isinstance(b, MoleculeTable)]
                if table_blocks:
                    self._replace_overlapping_blocks(
                        page,
                        table_blocks, 
                        self.table_overlap_threshold,
                        target_types=[BlockTypes.Table]
                    )

    def _replace_overlapping_blocks(self, page: PageGroup, new_blocks: List, 
                                   threshold: float, exclude_types: List = None, 
                                   target_types: List = None):
        """
        Replace overlapping blocks with new molecule/table blocks
        
        New logic: If any new_block overlaps with an existing_block above threshold,
        the existing_block will be removed. All new_blocks will be added.
        This handles cases where multiple molecules are within one figure.
        
        Args:
            page: The page containing blocks to check
            new_blocks: List of new blocks to add  
            threshold: Overlap threshold (0-1)
            exclude_types: Block types to exclude from replacement
            target_types: Only replace blocks of these types (if specified)
        """
        if not new_blocks:
            return
            
        if exclude_types is None:
            exclude_types = []
            
        blocks_to_remove = []  # existing blocks to remove
        blocks_to_add = new_blocks  # all new blocks will be added
        
        # First, identify all existing blocks that should be removed
        for existing_block in page.current_children:  # Use current_children to get non-removed blocks
            # Skip if block type is excluded
            if existing_block.block_type in exclude_types:
                continue
                
            # If target_types specified, only consider those types
            if target_types and existing_block.block_type not in target_types:
                continue
            
            # Check if this existing block overlaps with any new block above threshold
            should_remove = False
            for new_block in new_blocks:
                # Calculate overlap percentage (intersection area / new_block area)
                overlap_pct = new_block.polygon.intersection_pct(existing_block.polygon)
                print(f'overlap_pct: {overlap_pct:.3f} (intersection/new_block), existing_block: {existing_block.block_type}', flush=True)
                
                if overlap_pct >= threshold:
                    should_remove = True
                    print(f'🗑️  Will remove existing {existing_block.block_type} due to overlap {overlap_pct:.3f} with new molecule', flush=True)
                    break  # No need to check other new blocks for this existing block
                    
            if should_remove and existing_block not in blocks_to_remove:
                blocks_to_remove.append(existing_block)
        
        print(f'📊 Summary: Removing {len(blocks_to_remove)} existing blocks, Adding {len(blocks_to_add)} new blocks', flush=True)
        
        # Execute the operations
        self._execute_block_operations_v2(page, blocks_to_remove, blocks_to_add)

    def _execute_block_operations(self, page: PageGroup, blocks_to_replace: List, blocks_to_add: List):
        """
        Execute block replacement and addition operations using proper page methods
        
        Args:
            page: The page to modify
            blocks_to_replace: List of (old_block, new_block) tuples
            blocks_to_add: List of new blocks to add
        """
        # Replace existing blocks
        for old_block, new_block in blocks_to_replace:
            # Set proper page_id for the new block
            new_block.page_id = page.page_id
            page.replace_block(old_block, new_block)
        
        # Add new blocks
        for block_to_add in blocks_to_add:
            # Set proper page_id for the new block
            block_to_add.page_id = page.page_id
            page.add_full_block(block_to_add)
            # Also add to page structure for proper ordering
            page.structure.append(block_to_add.id)

    def _execute_block_operations_v2(self, page: PageGroup, blocks_to_remove: List, blocks_to_add: List):
        """
        Execute block removal and addition operations using proper page methods
        Maintains correct rendering order by inserting new blocks at removed blocks' positions
        
        Args:
            page: The page to modify
            blocks_to_remove: List of existing blocks to remove
            blocks_to_add: List of new blocks to add
        """
        if not blocks_to_remove and not blocks_to_add:
            return
            
        # Step 1: Record positions of blocks to be removed
        removal_positions = {}  # block_id -> position in structure
        if page.structure:
            for i, block_id in enumerate(page.structure):
                for block_to_remove in blocks_to_remove:
                    if block_id == block_to_remove.id:
                        removal_positions[block_to_remove.id] = i
                        break
        
        # Step 2: Remove existing blocks by marking them as removed
        for block_to_remove in blocks_to_remove:
            print(f'🔥 Removing existing block: {block_to_remove.block_type} at {block_to_remove.polygon.bbox}', flush=True)
            block_to_remove.removed = True
        
        # Step 3: Add new blocks and update structure with correct positioning
        for block_to_add in blocks_to_add:
            print(f'✅ Adding new block: {block_to_add.block_type} at {block_to_add.polygon.bbox}', flush=True)
            # Set proper page_id for the new block
            block_to_add.page_id = page.page_id
            page.add_full_block(block_to_add)
        
        # Step 4: Rebuild page structure with correct ordering
        if page.structure and removal_positions:
            # Find the earliest position where a block was removed
            earliest_position = min(removal_positions.values())
            print(f'📍 Inserting new blocks at position {earliest_position} (where removed blocks were)', flush=True)
            
            # Remove all removed block IDs from structure
            original_structure = page.structure[:]
            page.structure = [block_id for block_id in page.structure 
                            if not any(block_id == removed_block.id for removed_block in blocks_to_remove)]
            
            # Insert new block IDs at the earliest removal position
            new_block_ids = [block.id for block in blocks_to_add]
            page.structure[earliest_position:earliest_position] = new_block_ids
            
            print(f'🔄 Structure updated: {len(original_structure)} -> {len(page.structure)} blocks', flush=True)
        elif page.structure:
            # Fallback: append to end if no removal positions found
            for block_to_add in blocks_to_add:
                page.structure.append(block_to_add.id) 
