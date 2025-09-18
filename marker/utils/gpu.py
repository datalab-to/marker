import os
import subprocess
import torch

try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

from marker.logger import get_logger
from marker.settings import settings

logger = get_logger()


class GPUManager:
    default_gpu_vram: int = 8

    def __init__(self, device_idx: int):
        """
        Initialize the GPUManager.
        
        Args:
            device_idx (int): The index of the device to manage
        """
        self.device_idx = device_idx
        self.original_compute_mode = None
        self.mps_server_process = None

    def __enter__(self):
        """
        Context manager entry method.
        Initializes the appropriate device management system based on the device type.
        
        Returns:
            GPUManager: The current instance
        """
        if self.using_cuda():
            self.start_mps_server()
        elif self.using_xpu():
            self.initialize_xpu()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.
        Cleans up the appropriate device management system based on the device type.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if self.using_cuda():
            self.cleanup()
        elif self.using_xpu():
            self.cleanup_xpu()

    @staticmethod
    def using_cuda():
        """
        Check if CUDA device is being used.
        
        Returns:
            bool: True if CUDA device is being used, False otherwise
        """
        device_model = settings.TORCH_DEVICE_MODEL
        if callable(device_model):
            device_model = device_model()
        return "cuda" in device_model

    @staticmethod
    def using_xpu():
        """
        Check if XPU device is being used.
        
        Returns:
            bool: True if XPU device is being used, False otherwise
        """
        device_model = settings.TORCH_DEVICE_MODEL
        if callable(device_model):
            device_model = device_model()
        return "xpu" in device_model

    def check_xpu_available(self) -> bool:
        """
        Check if XPU devices are available and properly configured.
        
        Returns:
            bool: True if XPU is available, False otherwise
        """
        if not IPEX_AVAILABLE:
            logger.debug("Intel Extension for PyTorch (IPEX) not available")
            return False
            
        if not hasattr(torch, 'xpu'):
            logger.debug("torch.xpu module not available")
            return False
            
        if not torch.xpu.is_available():
            logger.debug("XPU devices not available")
            return False
            
        try:
            # Test basic XPU functionality
            x = torch.randn(1).to("xpu")
            del x
            logger.debug("XPU basic functionality test passed")
            return True
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"XPU basic functionality test failed: {e}")
            return False

    def check_xpu_ready(self) -> bool:
        """
        Check if XPU devices are not only available but also ready for use.
        This includes checking if the device is properly configured and has sufficient memory.
        
        Returns:
            bool: True if XPU is ready for use, False otherwise
        """
        if not self.check_xpu_available():
            return False
            
        try:
            # Check if we can allocate a small tensor
            x = torch.randn(1000, 1000, device="xpu")  # Allocate a 100x1000 tensor
            del x
            
            # Check memory availability
            memory_info = self.get_xpu_memory_info()
            if memory_info and 'free' in memory_info and memory_info['free'] < 100 * 1024 * 1024:  # Less than 10MB free
                logger.warning(f"XPU device {self.device_idx} has low memory: {memory_info['free'] / (1024**2):.2f} MB free")
                return False
                
            logger.debug(f"XPU device {self.device_idx} is ready for use")
            return True
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"XPU device {self.device_idx} not ready for use: {e}")
            return False

    def initialize_xpu(self) -> None:
        """
        Initialize XPU device for processing.
        This method can be used to set up any XPU-specific configurations.
        """
        if not self.check_xpu_available():
            return
            
        try:
            # Get device information for logging
            device_info = self.get_xpu_device_info()
            device_count = self.get_xpu_device_count()
            
            # Ensure the device is properly initialized
            # This is a simple check to ensure the device is working
            x = torch.randn(1, device="xpu")
            del x
            
            # Create a more informative log message
            device_name = device_info.get('name', 'Unknown') if device_info else 'Unknown'
            vram_info = f"{device_info.get('total_memory_gb', 'Unknown')} GB" if device_info and 'total_memory_gb' in device_info else 'Unknown GB'
            
            logger.info(f"Initialized XPU device {self.device_idx} ({device_name}) "
                       f"with {vram_info} VRAM. "
                       f"Total XPU devices: {device_count}")
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to initialize XPU device {self.device_idx}: {e}")

    def cleanup_xpu(self) -> None:
        """
        Cleanup XPU device resources.
        This method can be used to clean up any XPU-specific resources.
        """
        try:
            # Clear GPU cache if available
            if hasattr(torch.xpu, 'empty_cache'):
                torch.xpu.empty_cache()
            
            logger.info(f"Cleaned up XPU device {self.device_idx}")
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to clean up XPU device {self.device_idx}: {e}")

    def get_xpu_device_info(self) -> dict:
        """
        Get detailed information about the XPU device.
        
        Returns:
            dict: Dictionary containing device information including name, memory, etc.
        """
        if not self.check_xpu_available():
            return {}
            
        try:
            info = {}
            if hasattr(torch.xpu, 'get_device_name'):
                info['name'] = torch.xpu.get_device_name(self.device_idx)
                
            if hasattr(torch.xpu, 'get_device_properties'):
                props = torch.xpu.get_device_properties(self.device_idx)
                info['total_memory'] = getattr(props, 'total_memory', None)
                # Convert total memory to GB for readability
                if info['total_memory']:
                    info['total_memory_gb'] = int(info['total_memory'] / (1024 ** 3))
                
                # Add version information if available
                if hasattr(props, 'major'):
                    info['major'] = props.major
                if hasattr(props, 'minor'):
                    info['minor'] = props.minor
                
            logger.debug(f"XPU device {self.device_idx} info: {info}")
            return info
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get XPU device info: {e}")
            return {}

    def get_xpu_device_count(self) -> int:
        """
        Get the number of available XPU devices.
        
        Returns:
            int: Number of available XPU devices, 0 if XPU is not available
        """
        if not self.check_xpu_available():
            return 0
            
        try:
            if hasattr(torch.xpu, 'device_count'):
                count = torch.xpu.device_count()
                logger.debug(f"XPU device count: {count}")
                return count
            else:
                logger.debug("XPU device_count not available")
                return 0
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get XPU device count: {e}")
            return 0

    def check_xpu_capability(self, capability: str) -> bool:
        """
        Check if the XPU device supports a specific capability.
        
        Args:
            capability (str): The capability to check for (e.g., 'fp16', 'bf16', 'int8')
            
        Returns:
            bool: True if the capability is supported, False otherwise
        """
        if not self.check_xpu_available():
            return False
            
        try:
            # For now, we'll assume basic capabilities are supported
            # In the future, this could be expanded to check specific device capabilities
            supported_capabilities = ['fp32', 'fp16', 'bf16']
            result = capability in supported_capabilities
            logger.debug(f"XPU device {self.device_idx} capability '{capability}': {result}")
            return result
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to check XPU capability '{capability}': {e}")
            return False

    def get_xpu_memory_info(self) -> dict:
        """
        Get detailed memory information for the XPU device.
        
        Returns:
            dict: Dictionary containing memory information including allocated, reserved, etc.
        """
        if not self.check_xpu_available():
            return {}
            
        try:
            memory_info = {}
            
            # Get memory allocated by PyTorch
            if hasattr(torch.xpu, 'memory_allocated'):
                memory_info['allocated'] = torch.xpu.memory_allocated(self.device_idx)
                
            # Get memory reserved by PyTorch's memory allocator
            if hasattr(torch.xpu, 'memory_reserved'):
                memory_info['reserved'] = torch.xpu.memory_reserved(self.device_idx)
                
            # Get maximum memory allocated
            if hasattr(torch.xpu, 'max_memory_allocated'):
                memory_info['max_allocated'] = torch.xpu.max_memory_allocated(self.device_idx)
                
            # Get maximum memory reserved
            if hasattr(torch.xpu, 'max_memory_reserved'):
                memory_info['max_reserved'] = torch.xpu.max_memory_reserved(self.device_idx)
                
            # Get total and free memory
            if hasattr(torch.xpu, 'mem_get_info'):
                free, total = torch.xpu.mem_get_info(self.device_idx)
                memory_info['free'] = free
                memory_info['total'] = total
                
            logger.debug(f"XPU device {self.device_idx} memory info: {memory_info}")
            return memory_info
        except (RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to get XPU memory info: {e}")
            return {}

    def check_cuda_available(self) -> bool:
        """
        Check if CUDA devices are available and properly configured.
        
        Returns:
            bool: True if CUDA is available, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        try:
            subprocess.run(["nvidia-smi", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_xpu_vram(self) -> int:
        """
        Get XPU VRAM in GB.
        
        Returns:
            int: VRAM in GB, or default_gpu_vram if detection fails
        """
        if not self.check_xpu_available():
            return self.default_gpu_vram
            
        try:
            # Get XPU device properties
            if hasattr(torch.xpu, 'get_device_properties'):
                props = torch.xpu.get_device_properties(self.device_idx)
                # Total memory is in bytes, convert to GB
                total_memory_gb = int(props.total_memory / (1024 ** 3))
                logger.debug(f"XPU device {self.device_idx} VRAM: {total_memory_gb} GB")
                return max(1, total_memory_gb)  # Ensure at least 1GB
            else:
                # Fallback to a reasonable default for XPU
                logger.debug("XPU get_device_properties not available, using default VRAM")
                return self.default_gpu_vram
        except (RuntimeError, AttributeError, ValueError) as e:
            # If we can't get the memory info, return default
            logger.debug(f"Failed to get XPU VRAM info: {e}")
            return self.default_gpu_vram

    def get_gpu_vram(self):
        """
        Get GPU VRAM in GB based on the device type.
        
        Returns:
            int: VRAM in GB, or default_gpu_vram if detection fails
        """
        if self.using_cuda():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                        "-i",
                        str(self.device_idx),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                vram_mb = int(result.stdout.strip())
                vram_gb = int(vram_mb / 1024)
                return vram_gb

            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                return self.default_gpu_vram
        elif self.using_xpu():
            return self.get_xpu_vram()
        else:
            return self.default_gpu_vram

    def start_mps_server(self) -> bool:
        """
        Start the NVIDIA MPS (Multi-Process Service) server for CUDA devices.
        
        Returns:
            bool: True if MPS server started successfully, False otherwise
        """
        if not self.check_cuda_available():
            return False

        try:
            # Set MPS environment with chunk-specific directories
            env = os.environ.copy()
            pipe_dir = f"/tmp/nvidia-mps-{self.device_idx}"
            log_dir = f"/tmp/nvidia-log-{self.device_idx}"
            env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
            env["CUDA_MPS_LOG_DIRECTORY"] = log_dir

            # Create directories
            os.makedirs(pipe_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            # Start MPS control daemon
            self.mps_server_process = subprocess.Popen(
                ["nvidia-cuda-mps-control", "-d"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            logger.info(f"Started NVIDIA MPS server for chunk {self.device_idx}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                f"Failed to start MPS server for chunk {self.device_idx}: {e}"
            )
            return False

    def stop_mps_server(self) -> None:
        """
        Stop the NVIDIA MPS (Multi-Process Service) server for CUDA devices.
        """
        try:
            # Stop MPS server
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/nvidia-mps-{self.device_idx}"
            env["CUDA_MPS_LOG_DIRECTORY"] = f"/tmp/nvidia-log-{self.device_idx}"

            subprocess.run(
                ["nvidia-cuda-mps-control"],
                input="quit\n",
                text=True,
                env=env,
                timeout=10,
            )

            if self.mps_server_process:
                self.mps_server_process.terminate()
                try:
                    self.mps_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.mps_server_process.kill()
                self.mps_server_process = None

            logger.info(f"Stopped NVIDIA MPS server for chunk {self.device_idx}")
        except Exception as e:
            logger.warning(
                f"Failed to stop MPS server for chunk {self.device_idx}: {e}"
            )

    def cleanup(self) -> None:
        """
        Cleanup GPU resources.
        Stops the MPS server if it was started.
        """
        self.stop_mps_server()
