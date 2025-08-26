from __future__ import annotations

import asyncio
import logging
# import torch.multiprocessing as multiprocessing
# multiprocessing.set_start_method('spawn')
import multiprocessing
import torch
import tempfile
import json
try:
    torch.multiprocessing.set_start_method('spawn')
except:
    pass
import os
import signal
import sys
import threading
import traceback
from multiprocessing import Value
from typing import Any, Dict, List, NamedTuple

import aiohttp
import requests
import psutil
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from marker_main import ExtractionProc
from marker.utils import send_callback
import time
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv

load_dotenv()

# 获取北京时区
beijing_tz = pytz.timezone('Asia/Shanghai')

# 全局变量和锁
request_lock = threading.Lock()
stop_current_proc = Value("i", 0)

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 信号处理函数
def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}. Terminating all processes.")
    terminate_children(os.getpid())
    sys.exit(0)

# 终止所有子进程
def terminate_children(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            logging.info(f"Terminating process {child.pid}")
            try:
                child.terminate()
                child.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except psutil.NoSuchProcess:
        pass

# 任务队列和结果队列 - 拆分成两个队列
regular_queue = multiprocessing.Queue()  # 常规解析队列
molecule_queue = multiprocessing.Queue()  # 分子识别解析队列
result_queue = multiprocessing.Queue()

# 常规解析消费者进程
def regular_document_proc(
    regular_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    stop_current_proc: Value,
):
    print("init regular parsing model...", flush=True)
    extraction_proc = ExtractionProc()
    extraction_proc.load_models()
    print("regular parsing model inited", flush=True)

    while True:
        print("Regular queue: Waiting for new task...")
        params = regular_queue.get()
        if params is None:  # Shutdown signal
            break
        print("Starting regular extraction process")

        file = params.get("file", None)
        args = params.get("args", {})
        file_type = params.get("file_type", "pdf")
        docId = params.get("docId", "")
        callback_url = params.get("callback_url", "")
        skip_layout = params.get("skip_layout", False)
        print('callback_url', callback_url, flush=True)
        try:
            print('start>>>regular extraction', flush=True)
            print('file_type', file_type, type(file), flush=True)
            if file_type.lower() == "pdf":
                extraction_outputs = extraction_proc.extraction(
                    args, 
                    file, 
                    callback_url=callback_url,
                    docId=docId,
                    file_type=file_type,
                    mol_detect=False  # 强制关闭分子识别
                )
            elif file_type.lower() == "docx":
                # 直接转换DOCX到markdown，避免PDF转换的不准确性
                extraction_outputs = extraction_proc.parse_docx_direct(file)
            elif file_type.lower() == "pptx":
                # 将文件存储为临时文件，然后使用完整的extraction流程
                temp_file_path = os.path.join(tempfile.gettempdir(), f"{docId}.{file_type}")
                try:
                    with open(temp_file_path, "wb") as f:
                        f.write(file)
                    
                    extraction_outputs = extraction_proc.extraction(
                        args, 
                        temp_file_path, 
                        callback_url=callback_url,
                        docId=docId,
                        file_type=file_type,
                        mol_detect=False
                    )
                finally:
                    # 确保临时文件被清理
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            elif file_type == "jpg" or file_type == "png" or file_type == "jpeg":
                # 将file存在临时文件里面，并提供path
                temp_file_path = os.path.join(tempfile.gettempdir(), f"{docId}.{file_type}")
                with open(temp_file_path, "wb") as f:
                    f.write(file)
                
                # 对于图片输入，根据skip_layout参数决定是否跳过layout布局检测，强制OCR检测
                args['force_ocr'] = True
                args['force_layout_block'] = "Text"
                print(f"[ImageProcessing] skip_layout=True, forcing OCR and setting layout block to Text")

                extraction_outputs = extraction_proc.extraction(
                    args, 
                    temp_file_path, 
                    callback_url=callback_url,
                    docId=docId,
                    file_type=file_type
                )
            else:
                raise Exception("Unsupported file type")
            
            result_queue.put({"docId": docId, "result": extraction_outputs})
            if callback_url:
                time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
                
                # Handle dictionary format
                if isinstance(extraction_outputs, dict):
                    markdown_text = extraction_outputs.get('text', '')
                    metadata = extraction_outputs.get('metadata', {})
                    info = extraction_outputs.get('info', {})
                    images = extraction_outputs.get('images', {})
                    mol_images = extraction_outputs.get('mol_images', {})
                    table_contents = extraction_outputs.get('table_contents', {})
                    table_count = info.get('table_count', 0)
                    formula_count = info.get('formula_count', 0)
                    ocr_count = info.get('ocr_count', 0)
                else:
                    # Backward compatibility with tuple format
                    markdown_text = extraction_outputs[0] if len(extraction_outputs) > 0 else ''
                    metadata = extraction_outputs[3] if len(extraction_outputs) > 3 else {}
                    info = extraction_outputs[2] if len(extraction_outputs) > 2 else {}
                    images = {}
                    mol_images = {}
                    table_contents = {}
                    table_count = info.get('table_count', 0) if file_type == "pdf" else 0
                    formula_count = info.get('formula_count', 0) if file_type == "pdf" else 0
                    ocr_count = info.get('ocr_count', 0) if file_type == "pdf" else 0
                
                print('metadata', metadata, flush=True)
                send_callback(callback_url, {
                    'status': True,
                    'messages': 'success',
                    'markdown': markdown_text, 
                    'metadata': json.dumps(metadata),
                    'images': json.dumps(images),
                    'mol_images': json.dumps(mol_images),
                    'table_contents': json.dumps(table_contents),
                    'docId': docId,
                    'progress': 95,
                    'progress_text': '开始chunking和embedding\ntable数量 ' + str(table_count) + ' 公式数量 ' + str(formula_count) + ' ocr次数 ' + str(ocr_count) + '  ' + time_str
                })
        except Exception as e:
            traceback.print_exc()
            result_queue.put({
                "docId": docId, 
                "markdown": ' ', 
                "metadata": json.dumps({}),
                'images': json.dumps({}),
                'mol_images': json.dumps({}),
                'table_contents': json.dumps({}),
                'status': False,
                'messages': 'success'
            })
            if callback_url:
                time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
                send_callback(callback_url, {
                    'status': False,
                    'messages': 'error' + str(e),
                    'docId': docId,
                    'progress': 95,
                    'progress_text': 'error' + str(e)
                })
        finally:
            stop_current_proc.value = 0

# 分子识别消费者进程
def molecule_document_proc(
    molecule_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    stop_current_proc: Value,
):
    print("init molecule detection model...", flush=True)
    extraction_proc = ExtractionProc()
    extraction_proc.load_models()
    print("molecule detection model inited", flush=True)

    while True:
        print("Molecule queue: Waiting for new task...")
        params = molecule_queue.get()
        if params is None:  # Shutdown signal
            break
        print("Starting molecule detection extraction process")

        file = params.get("file", None)
        args = params.get("args", {})
        file_type = params.get("file_type", "pdf")
        docId = params.get("docId", "")
        callback_url = params.get("callback_url", "")
        skip_layout = params.get("skip_layout", False)
        print('callback_url', callback_url, flush=True)
        try:
            print('start>>>molecule detection extraction', flush=True)
            print('file_type', file_type, type(file), flush=True)
            if file_type.lower() == "pdf":
                extraction_outputs = extraction_proc.extraction(
                    args, 
                    file, 
                    callback_url=callback_url,
                    docId=docId,
                    file_type=file_type,
                    mol_detect=True  # 强制开启分子识别
                )
            elif file_type.lower() == "docx":
                # 直接转换DOCX到markdown，避免PDF转换的不准确性
                extraction_outputs = extraction_proc.parse_docx_direct(file)
            elif file_type.lower() == "pptx":
                # 将文件存储为临时文件，然后使用完整的extraction流程
                temp_file_path = os.path.join(tempfile.gettempdir(), f"{docId}.{file_type}")
                try:
                    with open(temp_file_path, "wb") as f:
                        f.write(file)
                    
                    extraction_outputs = extraction_proc.extraction(
                        args, 
                        temp_file_path, 
                        callback_url=callback_url,
                        docId=docId,
                        file_type=file_type,
                        mol_detect=True
                    )
                finally:
                    # 确保临时文件被清理
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
            elif file_type == "jpg" or file_type == "png" or file_type == "jpeg":
                # 将file存在临时文件里面，并提供path
                temp_file_path = os.path.join(tempfile.gettempdir(), f"{docId}.{file_type}")
                with open(temp_file_path, "wb") as f:
                    f.write(file)
                
                # 对于图片输入，根据skip_layout参数决定是否跳过layout布局检测，强制OCR检测
                args['force_ocr'] = True
                args['force_layout_block'] = "Text"
                print(f"[ImageProcessing] skip_layout=True, forcing OCR and setting layout block to Text")

                extraction_outputs = extraction_proc.extraction(
                    args, 
                    temp_file_path, 
                    callback_url=callback_url,
                    docId=docId,
                    file_type=file_type
                )
            else:
                raise Exception("Unsupported file type")
            
            result_queue.put({"docId": docId, "result": extraction_outputs})
            if callback_url:
                time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
                
                # Handle dictionary format
                if isinstance(extraction_outputs, dict):
                    markdown_text = extraction_outputs.get('text', '')
                    metadata = extraction_outputs.get('metadata', {})
                    info = extraction_outputs.get('info', {})
                    images = extraction_outputs.get('images', {})
                    mol_images = extraction_outputs.get('mol_images', {})
                    table_contents = extraction_outputs.get('table_contents', {})
                    table_count = info.get('table_count', 0)
                    formula_count = info.get('formula_count', 0)
                    ocr_count = info.get('ocr_count', 0)
                else:
                    # Backward compatibility with tuple format
                    markdown_text = extraction_outputs[0] if len(extraction_outputs) > 0 else ''
                    metadata = extraction_outputs[3] if len(extraction_outputs) > 3 else {}
                    info = extraction_outputs[2] if len(extraction_outputs) > 2 else {}
                    images = {}
                    mol_images = {}
                    table_contents = {}
                    table_count = info.get('table_count', 0) if file_type == "pdf" else 0
                    formula_count = info.get('formula_count', 0) if file_type == "pdf" else 0
                    ocr_count = info.get('ocr_count', 0) if file_type == "pdf" else 0
                
                print('metadata', metadata, flush=True)
                send_callback(callback_url, {
                    'status': True,
                    'messages': 'success',
                    'markdown': markdown_text, 
                    'metadata': json.dumps(metadata),
                    'images': json.dumps(images),
                    'mol_images': json.dumps(mol_images),
                    'table_contents': json.dumps(table_contents),
                    'docId': docId,
                    'progress': 95,
                    'progress_text': '开始chunking和embedding\ntable数量 ' + str(table_count) + ' 公式数量 ' + str(formula_count) + ' ocr次数 ' + str(ocr_count) + '  ' + time_str
                })
        except Exception as e:
            traceback.print_exc()
            result_queue.put({
                "docId": docId, 
                "markdown": ' ', 
                "metadata": json.dumps({}),
                'images': json.dumps({}),
                'mol_images': json.dumps({}),
                'table_contents': json.dumps({}),
                'status': False,
                'messages': 'success'
            })
            if callback_url:
                time_str = datetime.now(beijing_tz).strftime("%H:%M:%S")
                send_callback(callback_url, {
                    'status': False,
                    'messages': 'error' + str(e),
                    'docId': docId,
                    'progress': 95,
                    'progress_text': 'error' + str(e)
                })
        finally:
            stop_current_proc.value = 0


# 响应类
class ExtractionResponse(NamedTuple):
    status_code: int
    success: bool
    msg: str

# 生成响应
def do_response(resp: ExtractionResponse):
    return JSONResponse(resp._asdict())

# 处理文档提取请求
async def document_extract(request):
    try:
        form = await request.form()
        docId = form.get("docId", "")
        file_type = form.get("file_type", "pdf")  # pdf, docx, pptx, jpg, png, jpeg
        callback_url = form.get("callback_url", "")
        mol_detect = form.get("mol_detect", "False") == 'True'
        skip_layout = form.get("skip_layout", "False") == 'True'  # 新增参数：是否跳过layout检测
        print('mol_detect', form.get("mol_detect", ""), mol_detect, flush=True)
        print('skip_layout', form.get("skip_layout", ""), skip_layout, flush=True)

        if not docId:
            return do_response(
                ExtractionResponse(100, False, "No docId provided in request")
            )

        file = form.get("file", None)

        if file:
            file_content = await file.read()  # 读取文件内容
        else:
            file_content = None

        args = json.loads(form.get("args", "{}"))
        args['workers'] = 5
        
        extra = json.loads(form.get("extra", "{}"))
        is_testing = extra.get("is_testing", False)

        params = {
            "file": file_content,
            "file_type": file_type,
            "args": args,
            "docId": docId,
            "callback_url": callback_url,
            "mol_detect": mol_detect,
            "skip_layout": skip_layout,
        }

        if mol_detect:
            molecule_queue.put(params)
        else:
            regular_queue.put(params)

        return do_response(ExtractionResponse(200, True, "Task accepted"))
    except Exception as e:
        traceback.print_exc()
        return do_response(
            ExtractionResponse(102, False, f"Exception catch, cause {e!r}")
        )

# Ping响应
async def ping_resp(request):
    return JSONResponse({"response": "pong"})

# Starlette应用
app = Starlette(
    debug=True,
    routes=[
        Route("/api/v1/document_extract", document_extract, methods=["POST"]),
        Route("/api/ping", ping_resp, methods=["GET"]),
    ],
)

# 主函数
def main():
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动消费者进程
    processes = []
    
    # 启动4个常规解析消费者进程（处理速度快，需要更多并发）
    for _ in range(2):
        process = multiprocessing.Process(
            target=regular_document_proc,
            args=(
                regular_queue,
                result_queue,
                stop_current_proc,
            ),
        )
        process.start()
        processes.append(process)

    # 启动2个分子识别消费者进程（处理速度慢，占用资源多，减少并发数）
    for _ in range(1):
        process = multiprocessing.Process(
            target=molecule_document_proc,
            args=(
                molecule_queue,
                result_queue,
                stop_current_proc,
            ),
        )
        process.start()
        processes.append(process)

    # 启动Uvicorn服务器
    try:
        uvicorn.run(app, host=os.getenv("HOST"), port=int(os.getenv("PORT")))
    finally:
        # Shutdown consumers
        for _ in range(2):  # 关闭常规解析进程
            regular_queue.put(None)
        for _ in range(1):  # 关闭分子识别进程
            molecule_queue.put(None)
        for process in processes:
            process.join()

if __name__ == "__main__":
    main()
