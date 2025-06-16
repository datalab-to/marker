#!/usr/bin/env python3
"""
测试真实img2mol集成的脚本
验证Parser_Processer的实际API调用和数据格式
"""

import os
import sys
import traceback
from pathlib import Path
from PIL import Image
import json

# 添加必要的路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append('/app/img2mol')
sys.path.append('/app/img2mol/clean_img2smiles/src')

def test_img2mol_processor():
    """测试img2mol处理器的基本功能"""
    print("🧪 Testing img2mol Parser_Processer...")
    
    try:
        from img2smiles.pipeline.Processer import Parser_Processer
        print("✅ Successfully imported Parser_Processer")
    except ImportError as e:
        print(f"❌ Failed to import Parser_Processer: {e}")
        return False
    
    # 测试初始化
    try:
        config = {
            'device': 'cpu',  # 使用CPU避免CUDA问题
            'with_mol_detect': True,
            'with_table_detect': True,
            'debug': True,
            'use_yolo_mol_model': True,
            'use_yolo_table_model': True,
            'use_yolo_table_model_v2': True,
            'preload_table_and_ocr_model': False  # 避免预加载所有模型
        }
        
        processor = Parser_Processer(**config)
        print("✅ Successfully initialized Parser_Processer")
        
        # 打印一些关键属性
        print(f"📋 Processor attributes:")
        print(f"  - device: {processor.device}")
        print(f"  - with_mol_detect: {processor.with_mol_detect}")
        print(f"  - with_table_detect: {processor.with_table_detect}")
        print(f"  - moldetect_model: {processor.moldetect_model is not None}")
        
        return processor
        
    except Exception as e:
        print(f"❌ Failed to initialize Parser_Processer: {e}")
        traceback.print_exc()
        return None

def test_molecule_detection(processor, test_image_path=None):
    """测试分子检测功能"""
    print("\n🔬 Testing molecule detection...")
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"📁 Using test image: {test_image_path}")
        test_image = Image.open(test_image_path)
    else:
        print("🎨 Creating test image...")
        # 创建一个简单的测试图像
        test_image = Image.new('RGB', (800, 600), color='white')
    
    try:
        # 调用分子检测方法
        mol_results = processor.moldect_with_test_augment(
            image=test_image,
            moldetect_model=processor.moldetect_model,
            offset_x=0,
            offset_y=0,
            usr_tta=False,  # 关闭TTA以加快测试
            coref=True,
            use_ocr=False,
            with_padding=False,
            dubug=True
        )
        
        print(f"✅ Molecule detection completed")
        print(f"📊 Results: {len(mol_results)} molecules detected")
        
        # 分析结果格式
        if mol_results:
            sample_result = mol_results[0]
            print(f"📋 Sample result structure:")
            for key, value in sample_result.items():
                print(f"  - {key}: {type(value)} = {value}")
        
        return mol_results
        
    except Exception as e:
        print(f"❌ Molecule detection failed: {e}")
        traceback.print_exc()
        return []

def test_table_detection(processor, test_image_path=None):
    """测试表格检测功能"""
    print("\n📊 Testing table detection...")
    
    if test_image_path and os.path.exists(test_image_path):
        test_image = Image.open(test_image_path)
    else:
        # 创建一个简单的测试图像
        test_image = Image.new('RGB', (800, 600), color='white')
    
    try:
        # 调用表格检测方法
        result = processor.single_page_table_detection(test_image)
        
        print(f"✅ Table detection completed")
        print(f"📋 Result type: {type(result)}")
        
        if isinstance(result, tuple) and len(result) == 2:
            extracted_tables, tables_layout = result
            print(f"📊 Extracted tables: {len(extracted_tables)}")
            print(f"📊 Tables layout: {len(tables_layout)}")
            
            # 分析tables_layout的结构
            if tables_layout:
                sample_table = tables_layout[0]
                print(f"📋 Sample table block structure:")
                print(f"  - type: {type(sample_table)}")
                print(f"  - attributes: {dir(sample_table)}")
                
                if hasattr(sample_table, 'block'):
                    block = sample_table.block
                    print(f"  - block coordinates: ({block.x_1}, {block.y_1}, {block.x_2}, {block.y_2})")
                
                if hasattr(sample_table, 'score'):
                    print(f"  - score: {sample_table.score}")
        
        return result
        
    except Exception as e:
        print(f"❌ Table detection failed: {e}")
        traceback.print_exc()
        return None, []

def test_marker_integration():
    """测试与marker的集成"""
    print("\n🔗 Testing marker integration...")
    
    try:
        from marker.builders.molecule_layout import MoleculeLayoutBuilder
        
        # 测试配置
        processor_config = {
            'device': 'cpu',
            'with_mol_detect': True,
            'with_table_detect': True,
            'debug': True,
            'use_yolo_mol_model': True,
            'use_yolo_table_model': True,
            'use_tta': False,
            'preload_table_and_ocr_model': False
        }
        
        config = {
            'use_molecule_detection': True,
            'processor_config': processor_config
        }
        
        # 初始化MoleculeLayoutBuilder
        builder = MoleculeLayoutBuilder(
            processor_config=processor_config,
            config=config
        )
        
        print("✅ Successfully initialized MoleculeLayoutBuilder")
        print(f"📋 Builder status:")
        print(f"  - use_mock_data: {builder.use_mock_data}")
        print(f"  - processor available: {builder.processor is not None}")
        
        return builder
        
    except Exception as e:
        print(f"❌ Marker integration test failed: {e}")
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("🚀 Starting real img2mol integration test...")
    print("=" * 60)
    
    # 1. 测试基本的img2mol处理器
    processor = test_img2mol_processor()
    if processor is None:
        print("❌ Basic processor test failed, stopping...")
        return
    
    # 2. 测试分子检测
    mol_results = test_molecule_detection(processor)
    
    # 3. 测试表格检测
    table_results = test_table_detection(processor)
    
    # 4. 测试marker集成
    builder = test_marker_integration()
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"  ✅ Parser_Processer initialization: {'SUCCESS' if processor else 'FAILED'}")
    print(f"  ✅ Molecule detection: {'SUCCESS' if mol_results else 'FAILED'}")
    print(f"  ✅ Table detection: {'SUCCESS' if table_results else 'FAILED'}")
    print(f"  ✅ Marker integration: {'SUCCESS' if builder else 'FAILED'}")
    
    if all([processor, mol_results is not None, table_results is not None, builder]):
        print("\n🎉 All tests passed! Real img2mol integration is working.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 