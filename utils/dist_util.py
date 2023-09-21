"""
Helpers for distributed training.
"""
import paddle
import io
import os
import socket
import blobfile as bf
from mpi4py import MPI
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if paddle.distributed.is_initialized():
        return
    rank = MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE
    paddle.device.set_device(device=rank)
    comm = MPI.COMM_WORLD
    backend = 'gloo' if not paddle.device.cuda.device_count() >= 1 else 'nccl'
    if backend == 'gloo':
        hostname = 'localhost'
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ['MASTER_ADDR'] = comm.bcast(hostname, root=0)
    os.environ['RANK'] = str(comm.rank)
    os.environ['WORLD_SIZE'] = str(comm.size)
    port = comm.bcast(_find_free_port(), root=0)
    os.environ['MASTER_PORT'] = str(port)
    paddle.distributed.init_parallel_env()


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if paddle.device.cuda.device_count() >= 1:
        return str(f'cuda').replace('cuda', 'gpu')
    return str('cpu').replace('cuda', 'gpu')


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, 'rb') as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i:i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)
    return paddle.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with paddle.no_grad():
            paddle.distributed.broadcast(tensor=p, src=0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
