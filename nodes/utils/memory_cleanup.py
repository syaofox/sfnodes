import psutil
import ctypes
from ctypes import wintypes
import ctypes.util
import time
import platform
import gc
import comfy.model_management
from ...sf_utils.common import AnyType


any = AnyType("*")


class VRAMCleanup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "offload_model": ("BOOLEAN", {"default": True}),
                "offload_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "anything": (any, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "sfnodes/utils"
    DESCRIPTION = "清理 VRAM，卸载模型和清空缓存"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float(time.time())

    def empty_cache(
        self,
        offload_model,
        offload_cache,
        anything=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        try:
            if offload_model:
                comfy.model_management.unload_all_models()

            if offload_cache:
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()

            print(
                f"VRAM清理完成 [卸载模型: {offload_model}, 清空缓存: {offload_cache}]"
            )

        except Exception as e:
            print(f"VRAM清理失败: {str(e)}")

        return (anything,)


class RAMCleanup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clean_file_cache": (
                    "BOOLEAN",
                    {"default": True, "label": "清理文件缓存"},
                ),
                "clean_processes": (
                    "BOOLEAN",
                    {"default": True, "label": "清理进程内存"},
                ),
                "clean_buffers": ("BOOLEAN", {"default": True, "label": "清理系统缓冲区"}),
                "retry_times": (
                    "INT",
                    {"default": 3, "min": 1, "max": 10, "step": 1, "label": "重试次数"},
                ),
            },
            "optional": {
                "anything": (any, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "clean_ram"
    CATEGORY = "sfnodes/utils"
    DESCRIPTION = "清理系统 RAM，释放内存"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float(time.time())

    def get_ram_usage(self):
        memory = psutil.virtual_memory()
        return memory.percent, memory.available / (1024 * 1024)

    def clean_ram(
        self,
        clean_file_cache,
        clean_processes,
        clean_buffers,
        retry_times,
        anything=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        try:
            before_usage, before_available = self.get_ram_usage()
            system = platform.system()

            gc.collect()

            for attempt in range(retry_times):
                if clean_file_cache:
                    try:
                        if system == "Windows":
                            ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
                        elif system == "Linux":
                            lib_path = ctypes.util.find_library('c')
                            if lib_path:
                                libc = ctypes.CDLL(lib_path)
                                libc.malloc_trim(0)
                        elif system == "Darwin":
                            pass
                    except:
                        pass

                if clean_processes:
                    if system == "Windows":
                        for process in psutil.process_iter(["pid", "name"]):
                            try:
                                handle = ctypes.windll.kernel32.OpenProcess(
                                    wintypes.DWORD(0x001F0FFF),
                                    wintypes.BOOL(False),
                                    wintypes.DWORD(process.info["pid"]),
                                )
                                ctypes.windll.psapi.EmptyWorkingSet(handle)
                                ctypes.windll.kernel32.CloseHandle(handle)
                            except:
                                continue

                if clean_buffers:
                    try:
                        if system == "Windows":
                            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    except:
                        pass

                time.sleep(1)

            gc.collect()

            after_usage, after_available = self.get_ram_usage()
            freed_mb = after_available - before_available
            print(
                f"RAM清理完成 [{before_usage:.1f}% -> {after_usage:.1f}%, 释放: {freed_mb:.0f}MB]"
            )

        except Exception as e:
            print(f"RAM清理失败: {str(e)}")

        return (anything,)
