import ctypes
import numpy as np
from pathlib import Path

# serve per integrare la libreria C compilata come shared objects

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmetrics.so")))

_lib.metrics_count.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
]
_lib.metrics_count.restype = None

def metrics_count_np(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = np.ascontiguousarray(y_pred, dtype=np.uint8)
    y_true = np.ascontiguousarray(y_true, dtype=np.uint8)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

    N, K = y_true.shape

    tp = np.zeros((K,), dtype=np.uint64)
    tn = np.zeros((K,), dtype=np.uint64)
    fp = np.zeros((K,), dtype=np.uint64)
    fn = np.zeros((K,), dtype=np.uint64)

    _lib.metrics_count(
        y_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        y_true.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(N),
        ctypes.c_size_t(K),
        tp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        tn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        fp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        fn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )

    return tp, tn, fp, fn
