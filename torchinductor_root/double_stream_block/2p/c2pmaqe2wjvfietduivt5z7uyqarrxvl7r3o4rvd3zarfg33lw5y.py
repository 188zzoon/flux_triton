
# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_root/em/cemnfvm4jtfw37zlgr6zwhjjp7og6vxjdfjb7s75lzvl6rfs256q.py
# Source Nodes: [silu], Original ATen: [aten.silu]
# silu => convert_element_type, convert_element_type_1, mul, sigmoid
triton_poi_fused_silu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/bm/cbm7oiovz4glxugtjvql5lfvijpjoo5nbfpsyouho33cz6yuf6kb.py
# Source Nodes: [add, img_modulated, img_modulated_1, mul], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# add => add_1
# img_modulated => add, convert_element_type_10, convert_element_type_11, mul_2, rsqrt, sub, var_mean
# img_modulated_1 => add_2
# mul => mul_3
triton_red_fused_add_mul_native_layer_norm_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4080
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask & xmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp6 = 3072.0
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr1 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = 1.0
        tmp13 = tmp11 + tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp3
        tmp17 = tmp16 * tmp10
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 + tmp20
        tl.store(out_ptr1 + (r1 + (3072*x0)), tmp21, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/25/c25f2k5ywbboekf7xgjqanwa4abwpoonvya5wpolawd7qgltal53.py
# Source Nodes: [mean, pow_1, x], Original ATen: [aten._to_copy, aten.mean, aten.pow]
# mean => mean
# pow_1 => pow_1
# x => convert_element_type_15
triton_red_fused__to_copy_mean_pow_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_pow_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 97920
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x0) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xo/cxoffiqdvckzsg24upgfwvx2sk6t5lvtn36bk3yi3vngpdxkzafg.py
# Source Nodes: [add_2, mean, pow_1, rrms, x], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
# add_2 => add_3
# mean => mean
# pow_1 => pow_1
# rrms => rsqrt_1
# x => convert_element_type_15
triton_poi_fused__to_copy_add_mean_pow_rsqrt_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 4080
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (24*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 128.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x1 + (4096*y0)), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ln/cln22jj36vio7xijbw6s32kjabzwjw3ie6w6yh57i5g4rui47pl4.py
# Source Nodes: [mean_1, pow_2, x_1], Original ATen: [aten._to_copy, aten.mean, aten.pow]
# mean_1 => mean_1
# pow_2 => pow_2
# x_1 => convert_element_type_17
triton_red_fused__to_copy_mean_pow_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_pow_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 97920
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (3072 + r2 + (128*x0) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/xl/cxlbzhgqjshcyf6z3ekcay3eje4ezprhwjsdi66oszlpljqn6gxi.py
# Source Nodes: [add_4, mul_5, txt_modulated, txt_modulated_1], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# add_4 => add_6
# mul_5 => mul_9
# txt_modulated => add_5, convert_element_type_19, convert_element_type_20, mul_8, rsqrt_3, sub_1, var_mean_1
# txt_modulated_1 => add_7
triton_red_fused_add_mul_native_layer_norm_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask & xmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp6 = 3072.0
    tmp7 = tmp4 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp11 = tl.load(in_ptr1 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = 1.0
        tmp13 = tmp11 + tmp12
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp3
        tmp17 = tmp16 * tmp10
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 + tmp20
        tl.store(out_ptr1 + (r1 + (3072*x0)), tmp21, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/hq/chq7b3qfqfp43e3i4owk7zakcvxxymftjy4vexglpsldb3wnzlf3.py
# Source Nodes: [mean_2, pow_3, x_2], Original ATen: [aten._to_copy, aten.mean, aten.pow]
# mean_2 => mean_2
# pow_3 => pow_3
# x_2 => convert_element_type_24
triton_red_fused__to_copy_mean_pow_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_pow_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x0) + (9216*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/je/cje2us4ozxeedei23wic77r24zstc7abqn2q47wwkqzwv2zhs735.py
# Source Nodes: [add_6, mean_2, pow_3, rrms_2, x_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
# add_6 => add_8
# mean_2 => mean_2
# pow_3 => pow_3
# rrms_2 => rsqrt_4
# x_2 => convert_element_type_24
triton_poi_fused__to_copy_add_mean_pow_rsqrt_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 256
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (24*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 128.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-06
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/jr/cjrshq63qgxqwufmulphxi2tjpvuprak7pxofhk53q6antezutss.py
# Source Nodes: [mean_3, pow_4, x_3], Original ATen: [aten._to_copy, aten.mean, aten.pow]
# mean_3 => mean_3
# pow_4 => pow_4
# x_3 => convert_element_type_26
triton_red_fused__to_copy_mean_pow_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_pow_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (3072 + r2 + (128*x0) + (9216*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/4d/c4dlnzdob3fwdxfryz6uhikqmqmkzmzs564oamaeq4ziqlyw7lwd.py
# Source Nodes: [q_2], Original ATen: [aten.cat]
# q_2 => cat
triton_poi_fused_cat_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13320192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 4336
    x0 = xindex % 128
    x2 = (xindex // 555008)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (128*x2) + (9216*x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x1 + (256*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 4336, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr3 + ((-2359296) + x0 + (128*x2) + (9216*x1)), tmp14, other=0.0).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.load(in_ptr4 + ((-256) + x1 + (4096*x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (x0), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rj/crjve4m2r53d4ax3api7pzvbpb37dqpj4coqg35k53orpze4k3nl.py
# Source Nodes: [k_2], Original ATen: [aten.cat]
# k_2 => cat_1
triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13320192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 4336
    x0 = xindex % 128
    x2 = (xindex // 555008)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (3072 + x0 + (128*x2) + (9216*x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x1 + (256*x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 4336, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.load(in_ptr3 + ((-2356224) + x0 + (128*x2) + (9216*x1)), tmp14, other=0.0).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.load(in_ptr4 + ((-256) + x1 + (4096*x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr5 + (x0), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp14, tmp23, tmp24)
    tmp26 = tl.where(tmp4, tmp13, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/pm/cpmzo22axwwl2w3mstqcbpyiyruclykntmuv7fntjhjygvrbev4k.py
# Source Nodes: [v], Original ATen: [aten.cat]
# v => cat_2
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13320192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3072)
    x3 = xindex % 3072
    x0 = xindex % 128
    x1 = (xindex // 128) % 24
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (6144 + x3 + (9216*x2)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 4336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-2353152) + x3 + (9216*x2)), tmp8, other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (128*x2) + (555008*x1)), tmp14, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (18432, 3072), (3072, 1))
    assert_size_stride(primals_6, (18432, ), (1, ))
    assert_size_stride(primals_7, (18432, 3072), (3072, 1))
    assert_size_stride(primals_8, (18432, ), (1, ))
    assert_size_stride(primals_9, (9216, 3072), (3072, 1))
    assert_size_stride(primals_10, (9216, ), (1, ))
    assert_size_stride(primals_11, (9216, 3072), (3072, 1))
    assert_size_stride(primals_12, (9216, ), (1, ))
    assert_size_stride(primals_13, (1, 3072), (3072, 1))
    assert_size_stride(primals_14, (1, 4080, 3072), (12533760, 3072, 1))
    assert_size_stride(primals_15, (1, 256, 3072), (786432, 3072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [silu], Original ATen: [aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_silu_0.run(primals_13, buf0, 3072, grid=grid(3072), stream=stream0)
        del primals_13
        buf1 = empty_strided_cuda((1, 18432), (18432, 1), torch.bfloat16)
        # Source Nodes: [l__self___img_mod_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, buf0, reinterpret_tensor(primals_5, (3072, 18432), (1, 3072), 0), alpha=1, beta=1, out=buf1)
        del primals_5
        del primals_6
        buf2 = empty_strided_cuda((1, 18432), (18432, 1), torch.bfloat16)
        # Source Nodes: [l__self___txt_mod_lin], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, buf0, reinterpret_tensor(primals_7, (3072, 18432), (1, 3072), 0), alpha=1, beta=1, out=buf2)
        del primals_7
        del primals_8
        buf3 = empty_strided_cuda((1, 4080, 1), (4096, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 4080, 1), (4096, 1, 4096), torch.float32)
        buf6 = reinterpret_tensor(buf4, (1, 4080, 1), (4096, 1, 1), 0); del buf4  # reuse
        buf7 = empty_strided_cuda((1, 4080, 3072), (12533760, 3072, 1), torch.bfloat16)
        # Source Nodes: [add, img_modulated, img_modulated_1, mul], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_1.run(buf6, primals_14, buf1, buf3, buf7, 4080, 3072, grid=grid(4080), stream=stream0)
        buf8 = empty_strided_cuda((4080, 9216), (9216, 1), torch.bfloat16)
        # Source Nodes: [img_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf7, (4080, 3072), (3072, 1), 0), reinterpret_tensor(primals_9, (3072, 9216), (1, 3072), 0), alpha=1, beta=1, out=buf8)
        del primals_10
        buf9 = empty_strided_cuda((1, 24, 4080, 1), (97920, 1, 24, 97920), torch.float32)
        # Source Nodes: [mean, pow_1, x], Original ATen: [aten._to_copy, aten.mean, aten.pow]
        triton_red_fused__to_copy_mean_pow_2.run(buf8, buf9, 97920, 128, grid=grid(97920), stream=stream0)
        buf10 = empty_strided_cuda((1, 24, 4080, 1), (98304, 4096, 1, 1), torch.float32)
        # Source Nodes: [add_2, mean, pow_1, rrms, x], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_3.run(buf9, buf10, 24, 4080, grid=grid(24, 4080), stream=stream0)
        buf11 = buf9; del buf9  # reuse
        # Source Nodes: [mean_1, pow_2, x_1], Original ATen: [aten._to_copy, aten.mean, aten.pow]
        triton_red_fused__to_copy_mean_pow_4.run(buf8, buf11, 97920, 128, grid=grid(97920), stream=stream0)
        buf12 = empty_strided_cuda((1, 24, 4080, 1), (98304, 4096, 1, 1), torch.float32)
        # Source Nodes: [add_3, mean_1, pow_2, rrms_1, x_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_3.run(buf11, buf12, 24, 4080, grid=grid(24, 4080), stream=stream0)
        del buf11
        buf13 = empty_strided_cuda((1, 256, 1), (256, 1, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 256, 1), (256, 1, 256), torch.float32)
        buf16 = reinterpret_tensor(buf14, (1, 256, 1), (256, 1, 1), 0); del buf14  # reuse
        buf17 = empty_strided_cuda((1, 256, 3072), (786432, 3072, 1), torch.bfloat16)
        # Source Nodes: [add_4, mul_5, txt_modulated, txt_modulated_1], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_5.run(buf16, primals_15, buf2, buf13, buf17, 256, 3072, grid=grid(256), stream=stream0)
        buf18 = empty_strided_cuda((256, 9216), (9216, 1), torch.bfloat16)
        # Source Nodes: [txt_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf17, (256, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 9216), (1, 3072), 0), alpha=1, beta=1, out=buf18)
        del primals_12
        buf19 = empty_strided_cuda((1, 24, 256, 1), (6144, 1, 24, 6144), torch.float32)
        # Source Nodes: [mean_2, pow_3, x_2], Original ATen: [aten._to_copy, aten.mean, aten.pow]
        triton_red_fused__to_copy_mean_pow_6.run(buf18, buf19, 6144, 128, grid=grid(6144), stream=stream0)
        buf20 = empty_strided_cuda((1, 24, 256, 1), (6144, 256, 1, 1), torch.float32)
        # Source Nodes: [add_6, mean_2, pow_3, rrms_2, x_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_7.run(buf19, buf20, 24, 256, grid=grid(24, 256), stream=stream0)
        buf21 = buf19; del buf19  # reuse
        # Source Nodes: [mean_3, pow_4, x_3], Original ATen: [aten._to_copy, aten.mean, aten.pow]
        triton_red_fused__to_copy_mean_pow_8.run(buf18, buf21, 6144, 128, grid=grid(6144), stream=stream0)
        buf22 = empty_strided_cuda((1, 24, 256, 1), (6144, 256, 1, 1), torch.float32)
        # Source Nodes: [add_7, mean_3, pow_4, rrms_3, x_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.pow, aten.rsqrt]
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_7.run(buf21, buf22, 24, 256, grid=grid(24, 256), stream=stream0)
        del buf21
        buf23 = empty_strided_cuda((1, 24, 4336, 128), (13320192, 555008, 128, 1), torch.bfloat16)
        # Source Nodes: [q_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_9.run(buf18, buf20, primals_3, buf8, buf10, primals_1, buf23, 13320192, grid=grid(13320192), stream=stream0)
        buf24 = empty_strided_cuda((1, 24, 4336, 128), (13320192, 555008, 128, 1), torch.bfloat16)
        # Source Nodes: [k_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf18, buf22, primals_4, buf8, buf12, primals_2, buf24, 13320192, grid=grid(13320192), stream=stream0)
        buf25 = empty_strided_cuda((1, 24, 4336, 128), (13320192, 555008, 128, 1), torch.bfloat16)
        # Source Nodes: [v], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf18, buf8, buf25, 13320192, grid=grid(13320192), stream=stream0)
    return (buf23, buf24, buf25, reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 0), reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 3072), reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 6144), reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 9216), reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 12288), reinterpret_tensor(buf1, (1, 1, 3072), (18432, 18432, 1), 15360), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 0), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 3072), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 6144), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 9216), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 12288), reinterpret_tensor(buf2, (1, 1, 3072), (18432, 18432, 1), 15360), primals_1, primals_2, primals_3, primals_4, primals_14, primals_15, buf0, buf3, buf6, reinterpret_tensor(buf7, (4080, 3072), (3072, 1), 0), reinterpret_tensor(buf8, (1, 24, 4080, 128), (37601280, 128, 9216, 1), 0), reinterpret_tensor(buf8, (1, 24, 4080, 128), (37601280, 128, 9216, 1), 3072), buf10, buf12, buf13, buf16, reinterpret_tensor(buf17, (256, 3072), (3072, 1), 0), reinterpret_tensor(buf18, (1, 24, 256, 128), (2359296, 128, 9216, 1), 0), reinterpret_tensor(buf18, (1, 24, 256, 128), (2359296, 128, 9216, 1), 3072), buf20, buf22, reinterpret_tensor(primals_11, (9216, 3072), (3072, 1), 0), reinterpret_tensor(primals_9, (9216, 3072), (3072, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((1, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((1, 4080, 3072), (12533760, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((1, 256, 3072), (786432, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
