goal: add xpu device handling and usage to project. add new llm service llama-cpp and enable selection and usage for llm document processing in /config/parser
0) review marker directory in detail to confirm that items 1-4 are sufficient 
1) add llama-cpp python file to marker/services - edited for submission to llama.cpp server - tentatively complete
2) update gpu.py with xpu device handling class similar to cuda schema. should also include check_xpu_available similar to cuda, and start_mps_server should also check to see if xpu is available alongside cuda
3) update pyproject to have intel endpoint for torch and update poetry.lock
    python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
4) settings.py needs to be updated with handling for xpu device and automatic selection if available