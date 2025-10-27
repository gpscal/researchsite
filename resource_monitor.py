import threading
import time
import psutil


class ResourceMonitor(threading.Thread):
    """Monitor system resources and apply mitigations when thresholds are exceeded."""

    def __init__(self, ram_threshold: float = 0.95, gpu_threshold: float = 0.95, check_interval: int = 5):
        super().__init__(daemon=True)
        self.ram_threshold = ram_threshold
        self.gpu_threshold = gpu_threshold
        self.check_interval = check_interval
        self._running = True

    def run(self) -> None:
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            torch = None
            has_gpu = False

        while self._running:
            try:
                # RAM usage
                ram_usage = psutil.virtual_memory().percent / 100.0
                if ram_usage > self.ram_threshold:
                    self._clear_caches(torch)

                # GPU usage
                if has_gpu and torch is not None:
                    try:
                        total = torch.cuda.get_device_properties(0).total_memory
                        used = torch.cuda.memory_allocated(0)
                        if total > 0 and (used / total) > self.gpu_threshold:
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            finally:
                time.sleep(self.check_interval)

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _clear_caches(torch_mod) -> None:
        import gc
        gc.collect()
        if torch_mod is not None and hasattr(torch_mod, 'cuda') and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()


