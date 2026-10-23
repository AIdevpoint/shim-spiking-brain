"""Controllo preliminare: nessun server e nessuna simulazione avviati."""
import ctypes
from pathlib import Path
import platform
import sys

try:
    import torch
    import triton
except (ImportError, OSError) as exc:
    raise SystemExit(
        f"Dipendenza GPU assente: {exc}.\n"
        "Leggere LINUX.md; usare ./installa.sh --xpu per Intel o --cuda per NVIDIA."
    ) from exc

requested = sys.argv[1] if len(sys.argv) > 1 else None
cuda = torch.cuda.is_available()
xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
device = "cuda" if cuda else ("xpu" if xpu else None)
if device is None or (requested and device != requested):
    details = [f"GPU richiesta non disponibile a PyTorch {torch.__version__}."]
    if requested != "cuda":
        try:
            loader = ctypes.CDLL("libze_loader.so.1")
        except OSError as exc:
            details += [f"Level Zero non caricabile: {exc}",
                        "Runtime locale: ./installa.sh --runtime-intel"]
        else:
            loader.zeInit.argtypes = [ctypes.c_uint32]
            loader.zeInit.restype = ctypes.c_uint32
            result = loader.zeInit(0)
            details.append(f"Level Zero zeInit: 0x{result:08x}.")
            if result == 0:
                count = ctypes.c_uint32()
                loader.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
                loader.zeDriverGet.restype = ctypes.c_uint32
                result = loader.zeDriverGet(ctypes.byref(count), None)
                details.append(f"Driver Level Zero: {count.value}, risultato 0x{result:08x}.")
            details.append("Verificare compatibilita del driver Intel Windows e del runtime WSL; vedere LINUX.md.")
        if "microsoft" in platform.release().lower():
            details.append("Dispositivo WSL /dev/dxg: " + ("presente" if Path("/dev/dxg").exists() else "non visibile in questa sessione"))
    raise SystemExit("\n".join(details))

try:
    value = torch.ones(1, device=device)
    if (value + 1).item() != 2:
        raise RuntimeError("operazione GPU non riuscita")
    target = triton.runtime.driver.active.get_current_target()
except Exception as exc:
    raise SystemExit(f"GPU/Triton non utilizzabile: {exc}. Consultare LINUX.md.") from exc
print(f"PyTorch {torch.__version__}, Triton {triton.__version__}: {device} ({target.backend})")
