"""Check caller device preservation using the public C communication API."""
import ctypes as c
import os
from pathlib import Path
import torch

library=c.CDLL(str(Path(os.environ['INFINI_ROOT'])/'lib/libinfiniccl.so'))
library.infinicclCommInitAll.argtypes=[c.c_int,c.POINTER(c.c_void_p),c.c_int,c.POINTER(c.c_int)]
library.infinicclCommInitAll.restype=c.c_int
library.infinicclCommDestroy.argtypes=[c.c_void_p]
library.infinicclCommDestroy.restype=c.c_int
for current in (0,1):
    comms=(c.c_void_p*2)()
    devices=(c.c_int*2)(0,1)
    assert library.infinicclCommInitAll(1,comms,2,devices)==0
    torch.cuda.set_device(current)
    for comm in comms:
        assert library.infinicclCommDestroy(comm)==0
        assert torch.cuda.current_device()==current
print('PASS: 2 TP2 communicator groups; caller device 0/1 preserved after every destroy')
