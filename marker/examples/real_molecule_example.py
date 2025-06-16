#!/usr/bin/env python3
"""
真实img2mol集成使用示例
展示如何正确配置和使用真实的分子识别功能
"""

import os
import sys
from pathlib import Path

# 添加marker路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from marker.main import marker_main
from marker.config.parser import ConfigParser
from marker.config.processor import apply_config

def create_real_molecule_config():
    """创建真实img2mol处理的配置"""
    
    # img2mol处理器配置
    processor_config = {
        # 基础设置
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 优先使用GPU
        'debug': True,  # 开启调试模式
        'num_workers': 1,  # 工作线程数
        'padding': 0,  # 图像填充
        
        # 分子检测设置
        'with_mol_detect': True,  # 启用分子检测
        'use_yolo_mol_model': True,  # 使用YOLO分子模型
        'new_class_token': True,  # 使用新的类标记
        
        # 表格检测设置  
        'with_table_detect': True,  # 启用表格检测
        'use_yolo_table_model': True,  # 使用YOLO表格模型
        'use_yolo_table_model_v2': True,  # 使用YOLO表格模型v2
        
        # OCR设置
        'use_trocr_mfr_model_v3': True,  # 使用TrOCR数学公式识别模型v3
        'use_got_ocr_model': True,  # 使用GOT OCR模型
        'with_ocr': False,  # 在分子检测时不使用OCR
        
        # 模型预加载设置（可选，影响启动速度）
        'preload_table_and_ocr_model': True,  # 预加载表格和OCR模型
        
        # 高级设置
        'use_tta': True,  # 使用测试时间增强
        'coref': True,  # 使用共指消解
        'with_padding': False,  # 不使用图像填充
        
        # 模型路径（如果需要指定特定模型）
        # 'MolDetect_mol_path': '/path/to/mol/model',
        # 'model_dir': '/path/to/models'
    }
    
    # marker主要配置
    config = {
        # 启用分子检测
        'use_molecule_detection': True,
        
        # 传递处理器配置
        'processor_config': processor_config,
        
        # 其他marker配置
        'max_pages': None,  # 处理所有页面
        'start_page': None,  # 从第一页开始
        'languages': ['en'],  # 支持的语言
        'batch_multiplier': 1,  # 批处理倍数
        'ocr_all_pages': False,  # 不对所有页面进行OCR
        
        # 输出格式
        'output_format': 'markdown',  # 输出格式为markdown
        'extract_images': True,  # 提取图像
    }
    
    return config

def process_chemical_pdf(pdf_path, output_dir=None):
    """
    处理化学PDF文档，检测分子和表格
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录，如果为None则使用默认目录
    
    Returns:
        处理结果的路径
    """
    
    # 验证输入文件
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_molecule_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"🧬 开始处理化学PDF: {pdf_path}")
    print(f"📁 输出目录: {output_dir}")
    
    # 创建配置
    config = create_real_molecule_config()
    
    try:
        # 调用marker主函数
        result = marker_main(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            config=config,
            artifact_dict={
                'processor_config': config['processor_config']  # 确保传递处理器配置
            }
        )
        
        print(f"✅ 处理完成!")
        print(f"📄 输出文件: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用真实img2mol处理化学PDF文档')
    parser.add_argument('pdf_path', help='输入PDF文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录路径')
    parser.add_argument('--mock', action='store_true', help='使用mock模式（测试用）')
    
    args = parser.parse_args()
    
    # 如果使用mock模式，修改配置
    if args.mock:
        print("🎭 使用Mock模式进行测试")
        config = create_real_molecule_config()
        config['processor_config']['use_mock_data'] = True
    
    # 处理PDF
    result = process_chemical_pdf(args.pdf_path, args.output_dir)
    
    if result:
        print(f"\n🎉 处理成功完成!")
        print(f"📋 结果文件: {result}")
        print(f"\n💡 提示: 查看输出的markdown文件，其中包含:")
        print(f"   • <mol>...</mol> 标签标记的分子结构")
        print(f"   • <mol_table>...</mol_table> 标签标记的分子表格")
    else:
        print(f"\n❌ 处理失败，请检查错误消息")
        sys.exit(1)

if __name__ == "__main__":
    # 导入必要的模块
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not available, defaulting to CPU")
        torch = None
    
    main() 