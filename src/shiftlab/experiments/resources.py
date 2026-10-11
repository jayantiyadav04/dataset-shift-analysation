"""OS-measured memory without an extra monitoring dependency."""

import ctypes
import os
import platform


def resource_measurement() -> dict:
    if os.name != "nt":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "peak_process_ram_bytes": int(peak * (1 if platform.system() == "Darwin" else 1024)),
            "definition": "OS process-lifetime peak resident set",
        }

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Counters),
        ctypes.c_ulong,
    ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {
        "peak_process_ram_bytes": int(counters.PeakWorkingSetSize),
        "peak_process_commit_bytes": int(counters.PeakPagefileUsage),
        "definition": "Windows process-lifetime peak working set; this single process includes "
        "imports, preprocessing, CV, all estimators and persistence; no GPU memory",
    }


def hardware_report() -> dict:
    result = {
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu": platform.processor(),
        "user_reported": {"cpu": "Intel i7-14700HX", "ram_gb": 16, "gpu": "RTX 4060 8 GB"},
        "execution": "CPU only; one job and one numeric thread; GPU unused",
    }
    if os.name == "nt":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            result["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("GlobalMemoryStatusEx failed")
        result.update(
            {
                "physical_ram_bytes": int(memory.total),
                "available_ram_bytes_at_measurement": int(memory.available),
            }
        )
    return result
