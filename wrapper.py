import ctypes
import numpy as np
from pathlib import Path

# serve per integrare la libreria C compilata come shared objects

_lib = ctypes.CDLL(str(Path(__file__).with_name("libmetrics.so")))

# qua definisco la funzione C e i suoi argomenti e tipi di ritorno, il Python non sa quali tipi di dato aspettarsi quindi devo specificarlo
# con lo stesso ordine degli argomenti della funzione C
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
# questo mi dice che la funzione non ritorna nulla (void), i risultati vengono passati tramite i puntatori che passo alla funzione
_lib.metrics_count.restype = None

# questo ascontiguousarray è fondamentale perchè assicura che gli array numpy siano contigui in memoria, altrimenti la libreria C potrebbe non funzionare correttamente
# siccome C si aspetta che i dati siano in un blocco contiguo di memoria
def metrics_count_np(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = np.ascontiguousarray(y_pred, dtype=np.uint8)
    y_true = np.ascontiguousarray(y_true, dtype=np.uint8)

    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

    N, K = y_true.shape

    # qui alloco la memoria RAM che sarà riempita dalla funzione C con i conteggi di true positive, true negative, false positive e false negative per ciascuna classe
    # qui è memoria nell'heap del processo Python, creo un treno continuo di memoria per ciascun array di 8 byte (uint64) per K 
    tp = np.zeros((K,), dtype=np.uint64)
    tn = np.zeros((K,), dtype=np.uint64)
    fp = np.zeros((K,), dtype=np.uint64)
    fn = np.zeros((K,), dtype=np.uint64)

    _lib.metrics_count(
        y_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        y_true.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(N),
        ctypes.c_size_t(K),
        # passo i puntatori agli array che ho allocato sopra, l'indirizzo intero del primo byte del buffer.
        tp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        tn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        fp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        fn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )

    return tp, tn, fp, fn
