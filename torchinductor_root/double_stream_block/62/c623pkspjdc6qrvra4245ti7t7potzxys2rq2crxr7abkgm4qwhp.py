
# AOT ID: ['1_forward']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/fn/cfnxw25ku6f4xek7geujgyenpdwixbuuftnucjjyznaandm4y6q4.py
# Source Nodes: [cos_sin, neg_sin_cos], Original ATen: [aten.stack]
# cos_sin => cat
# neg_sin_cos => cat_1
triton_poi_fused_stack_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 555008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (x1), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tl.load(in_ptr0 + (x1), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp18, tmp21)
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp22, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/gs/cgs2gzow2um6uvej2bwh6wfffzcmw6jwfccpdm7pxddfopkyd4ij.py
# Source Nodes: [xk_out_1, xq_out_1], Original ATen: [aten._to_copy]
# xk_out_1 => convert_element_type_3
# xq_out_1 => convert_element_type_2
triton_poi_fused__to_copy_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13320192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % 555008
    x0 = xindex % 128
    x4 = (xindex // 128)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((2*(x0 // 2)) + (128*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1 + (2*(x0 // 2)) + (128*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr3 + ((2*(x0 // 2)) + (128*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (1 + (2*(x0 // 2)) + (128*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp1 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp6 * tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp11, None)
    tl.store(out_ptr1 + (x5), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xh/cxhn27nzdi7b65zfzc7slaufelnq5eobyq4swvhxd5teuulveeeu.py
# Source Nodes: [x_1], Original ATen: [aten.clone]
# x_1 => clone
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13320192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 24
    x2 = (xindex // 3072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (555008*x1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, 24, 4336, 128), (13320192, 555008, 128, 1))
    assert_size_stride(primals_2, (1, 4336, 64), (277504, 64, 1))
    assert_size_stride(primals_3, (1, 4336, 64), (277504, 64, 1))
    assert_size_stride(primals_4, (1, 24, 4336, 128), (13320192, 555008, 128, 1))
    assert_size_stride(primals_5, (1, 24, 4336, 128), (13320192, 555008, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 4336, 64, 2), (555008, 555008, 128, 2, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((1, 1, 4336, 64, 2), (555008, 555008, 128, 2, 1), torch.bfloat16)
        # Source Nodes: [cos_sin, neg_sin_cos], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_0.run(primals_2, primals_3, buf0, buf1, 555008, grid=grid(555008), stream=stream0)
        del primals_2
        del primals_3
        buf2 = empty_strided_cuda((1, 24, 4336, 128), (13320192, 555008, 128, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((1, 24, 4336, 128), (13320192, 555008, 128, 1), torch.bfloat16)
        # Source Nodes: [xk_out_1, xq_out_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_1.run(buf0, primals_1, buf1, primals_4, buf2, buf3, 13320192, grid=grid(13320192), stream=stream0)
        del primals_1
        del primals_4
        # Source Nodes: [x], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf4 = aten._scaled_dot_product_flash_attention.default(buf2, buf3, primals_5, scale=0.08838834764831843)
        buf5 = buf4[0]
        buf6 = buf4[1]
        buf7 = buf4[6]
        buf8 = buf4[7]
        del buf4
        buf10 = empty_strided_cuda((1, 4336, 24, 128), (13320192, 3072, 128, 1), torch.bfloat16)
        # Source Nodes: [x_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf5, buf10, 13320192, grid=grid(13320192), stream=stream0)
    return (reinterpret_tensor(buf10, (1, 4336, 3072), (13320192, 3072, 1), 0), primals_5, buf0, buf1, buf2, buf3, buf5, buf6, buf7, buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 24, 4336, 128), (13320192, 555008, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1, 4336, 64), (277504, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 4336, 64), (277504, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 24, 4336, 128), (13320192, 555008, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((1, 24, 4336, 128), (13320192, 555008, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
