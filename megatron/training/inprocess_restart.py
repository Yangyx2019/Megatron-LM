# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib
import os
import socket
import warnings
from datetime import timedelta

import torch

from megatron.core import rerun_state_machine
from megatron.training import get_args
from megatron.training.async_utils import (
    reset_persistent_async_worker,
)

from . import arguments


def _get_inprocess_module():
    try:
        """
            importlib.import_module是运行时动态导入，而import语句是在编译时静态导入
            todo: 回头再来看nvidia_resiliency_ext.inprocess
        """
        return importlib.import_module("nvidia_resiliency_ext.inprocess")
    except ImportError:
        return None


def destroy_state():
    from . import training
    training.destroy_global_state()
    rerun_state_machine.destroy_rerun_state_machine()

def inprocess_restart(train, args):
    inprocess = _get_inprocess_module()
    if inprocess is None:
        warnings.warn('In-process restart is not available')
        return train

    if 'TORCH_CPP_LOG_LEVEL' not in os.environ or os.environ['TORCH_CPP_LOG_LEVEL'] not in (
        'error',
        'fatal',
    ):
        warnings.warn(
            'Set TORCH_CPP_LOG_LEVEL=error to suppress c10d waitForInput timeout warning messages'
        )

    # Layers represents a configuration for a layer of branches at a certain
    # depth in a topology tree constructed by inprocess.rank_assignment.Tree.
    # First layer contains all ranks and it's the root of the topology tree,
    # the second optional layer groups ranks by nodes.
    layers = [
        inprocess.rank_assignment.Layer(
            min_ranks=args.inprocess_active_world_size,
            max_ranks=args.inprocess_active_world_size,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE,
        )
    ]
    if args.inprocess_granularity == 'node':
        device_count = torch.cuda.device_count()

        layers.append(
            inprocess.rank_assignment.Layer(
                min_ranks=device_count,
                max_ranks=device_count,
                key_or_fn=lambda _: socket.gethostname(),
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            )
        )

    finalize = [
        inprocess.finalize.ThreadedFinalize(timeout=timedelta(seconds=10), fn=destroy_state)
    ]

    if args.inprocess_empty_cuda_cache:
        finalize.append(
            inprocess.finalize.ThreadedFinalize(
                timeout=timedelta(seconds=10), fn=torch.cuda.empty_cache
            )
        )

    initialize = inprocess.Compose(
        inprocess.initialize.RetryController(min_world_size=args.inprocess_active_world_size),
        inprocess.nested_restarter.NestedRestarterHandlingCompleted(),
    )

    class AbortCheckpoint(inprocess.abort.Abort):
        def __init__(self, async_strategy):
            self.async_strategy = async_strategy
        def __call__(
            self, state: inprocess.state.FrozenState
        ) -> inprocess.state.FrozenState:
            reset_persistent_async_worker(self.async_strategy)
            return state

    abort = inprocess.Compose(
        inprocess.abort.AbortTransformerEngine(),
        inprocess.abort.AbortTorchDistributed(),
        AbortCheckpoint(args.async_strategy),
        inprocess.nested_restarter.NestedRestarterHandlingStarting(),
    )
    completion = inprocess.nested_restarter.NestedRestarterFinalized()
    terminate = inprocess.nested_restarter.NestedRestarterAborted()

    train = inprocess.Wrapper(
        store_kwargs={
            'timeout': timedelta(seconds=300),
            'port': int(os.environ['MASTER_PORT']) + 2,
        },
        initialize=initialize,
        abort=abort,
        completion=completion,
        terminate=terminate,
        health_check=inprocess.health_check.CudaHealthCheck(timeout=timedelta(seconds=10)),
        rank_assignment=inprocess.rank_assignment.Tree(layers=layers),
        finalize=inprocess.Compose(*finalize),
        heartbeat_interval=timedelta(seconds=args.inprocess_heartbeat_interval),
        heartbeat_timeout=timedelta(seconds=args.inprocess_heartbeat_timeout),
        barrier_timeout=timedelta(seconds=args.inprocess_barrier_timeout),
        completion_timeout=timedelta(seconds=args.inprocess_completion_timeout),
        monitor_process_interval=timedelta(seconds=args.inprocess_monitor_process_interval),
        monitor_thread_interval=timedelta(seconds=args.inprocess_monitor_thread_interval),
        last_call_wait=timedelta(seconds=args.inprocess_last_call_wait),
        soft_timeout=timedelta(seconds=args.inprocess_soft_timeout),
        hard_timeout=timedelta(seconds=args.inprocess_hard_timeout),
        termination_grace_time=timedelta(seconds=args.inprocess_termination_grace_time),
        enabled=True,
    )(train)

    return train


def maybe_wrap_for_inprocess_restart(pretrain):

    args = arguments.parse_args(ignore_unknown_args=True)
    """
        inprocess_restart参数的意义：
            如果为True，则启用inprocess_restart功能，在训练过程中如果检测到某些错误（如节点故障），
            可以自动重启训练过程，而不需要手动干预。这对于分布式训练非常有用，可以提高训练的鲁棒性和效率。
            具体来说，当inprocess_restart启用时，训练过程会被包装在一个inprocess_restart的上下文中，
            这个上下文会监控训练过程中的状态，并在检测到错误时自动重启训练过程。
            重启过程中会执行一些清理操作，如销毁当前的训练状态，重置一些变量等，
            以确保重启后的训练过程能够正确地继续进行。

            比直接从checkpoint重新拉起训练快。
    """
    if args.inprocess_restart:
        pretrain = inprocess_restart(pretrain, args)
        """
        torch中用于tcp通信的工具。每个进程都调用这个api，rank0作为master，其他rank作为worker，进行通信同步。
        如果其他rank调用了这个api但是rank0没有调用，那么其他rank会一直等待，直到rank0调用了这个api。
        eg:

            import torch.distributed as dist
            from datetime import timedelta
            # Run on process 1 (server)
            server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
            # Run on process 2 (client)
            client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
            # Use any of the store methods from either the client or server after initialization
            server_store.set("first_key", "first_value")
            client_store.get("first_key")
        """
        store = torch.distributed.TCPStore(
            host_name=os.environ['MASTER_ADDR'],
            port=int(os.environ['MASTER_PORT'])+1,
            world_size=int(os.getenv('WORLD_SIZE', '1')),
            is_master=(int(os.getenv('RANK', '0')) == 0),
            timeout=timedelta(seconds=300),
            wait_for_workers=True,
            use_libuv=True,
        )
    else:
        store = None

    return pretrain, store


def maybe_force_nccl_backend_init(device_id):

    args = get_args()

    # Inprocess uses destroy_process_group to terminate NCCL backend, which
    # does not terminate NCCL kernels if NCCL backend wasn't fully initialized
    # before additional distributed subgroups are created. This forces initialization
    # of the NCCL backend.
    if args.inprocess_restart:
        tensor = torch.ones(128, device=device_id)
        torch.distributed.all_reduce(tensor)
        torch.cuda.synchronize()
