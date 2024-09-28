
# AOT ID: ['2_forward']
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


# kernel path: /tmp/torchinductor_root/be/cbeewuonzimws5fbcpigm4bjdgshaxmovlscnnzgwv443vdhjxkg.py
# Source Nodes: [add_1, add_2, img, l__self___img_norm2, mul, mul_1], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# add_1 => add_1
# add_2 => add_3
# img => add
# l__self___img_norm2 => add_2, convert_element_type_3, convert_element_type_4, mul_1, rsqrt, sub, var_mean
# mul => mul
# mul_1 => mul_2
triton_red_fused_add_mul_native_layer_norm_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4080
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp10 = 3072.0
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp21 = tmp19 * tmp20
        tmp22 = tmp18 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp7
        tmp25 = tmp24 * tmp14
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp17 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr1 + (r1 + (3072*x0)), tmp29, rmask & xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# kernel path: /tmp/torchinductor_root/bz/cbzdhy2zk5picgn2jqv6z3gmrbncxrl4qxdwobhqhmjcf3ateweu.py
# Source Nodes: [l__self___img_mlp_1], Original ATen: [aten.gelu]
# l__self___img_mlp_1 => add_4, add_5, convert_element_type_8, convert_element_type_9, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, tanh
triton_poi_fused_gelu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50135040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/2w/c2wu4fcdd52q4mn5upyd5lxwbwpxilmfcjl7but6mclm6dswjmhd.py
# Source Nodes: [img, img_1, mul, mul_2], Original ATen: [aten.add, aten.mul]
# img => add
# img_1 => add_6
# mul => mul
# mul_2 => mul_9
triton_poi_fused_add_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12533760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/vw/cvwnb5dttkgtup3jr52u6kflp3prsjzs2d2miz5zymkprmdz3iuj.py
# Source Nodes: [add_5, add_6, l__self___txt_norm2, mul_3, mul_4, txt], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
# add_5 => add_8
# add_6 => add_10
# l__self___txt_norm2 => add_9, convert_element_type_16, convert_element_type_17, mul_11, rsqrt_1, sub_1, var_mean_1
# mul_3 => mul_10
# mul_4 => mul_12
# txt => add_7
triton_red_fused_add_mul_native_layer_norm_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
    tmp10 = 3072.0
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = 1.0
        tmp17 = tmp15 + tmp16
        tmp21 = tmp19 * tmp20
        tmp22 = tmp18 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp23 - tmp7
        tmp25 = tmp24 * tmp14
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp17 * tmp26
        tmp29 = tmp27 + tmp28
        tl.store(out_ptr1 + (r1 + (3072*x0)), tmp29, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/qq/cqq7lbe52ajs4auzzk5g7i6k7ecznwpj7cb4igb35enhue3rjp4d.py
# Source Nodes: [l__self___txt_mlp_1], Original ATen: [aten.gelu]
# l__self___txt_mlp_1 => add_11, add_12, convert_element_type_21, convert_element_type_22, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, tanh_1
triton_poi_fused_gelu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/kf/ckfzaya4aw6l5blx3wiuda4j7kutg6g7iji474qsmvvgcz3ecuby.py
# Source Nodes: [mul_3, mul_5, txt, txt_1], Original ATen: [aten.add, aten.mul]
# mul_3 => mul_10
# mul_5 => mul_19
# txt => add_7
# txt_1 => add_13
triton_poi_fused_add_mul_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=84), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'E47CFD65927A6C9969DAEDAC9885EEDF57B12BE19313F83297707B3EFCB1B3CC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23 = args
    args.clear()
    assert_size_stride(primals_1, (3072, 3072), (3072, 1))
    assert_size_stride(primals_2, (3072, ), (1, ))
    assert_size_stride(primals_3, (12288, 3072), (3072, 1))
    assert_size_stride(primals_4, (12288, ), (1, ))
    assert_size_stride(primals_5, (3072, 12288), (12288, 1))
    assert_size_stride(primals_6, (3072, ), (1, ))
    assert_size_stride(primals_7, (3072, 3072), (3072, 1))
    assert_size_stride(primals_8, (3072, ), (1, ))
    assert_size_stride(primals_9, (12288, 3072), (3072, 1))
    assert_size_stride(primals_10, (12288, ), (1, ))
    assert_size_stride(primals_11, (3072, 12288), (12288, 1))
    assert_size_stride(primals_12, (3072, ), (1, ))
    assert_size_stride(primals_13, (1, 4336, 3072), (13320192, 3072, 1))
    assert_size_stride(primals_14, (1, 256, 3072), (786432, 3072, 1))
    assert_size_stride(primals_15, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_16, (1, 4080, 3072), (12533760, 3072, 1))
    assert_size_stride(primals_17, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_18, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_19, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_20, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_21, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_22, (1, 1, 3072), (3072, 3072, 1))
    assert_size_stride(primals_23, (1, 1, 3072), (3072, 3072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4080, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [l__self___img_attn_proj], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_13, (4080, 3072), (3072, 1), 786432), reinterpret_tensor(primals_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((1, 4080, 1), (4096, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 4080, 1), (4096, 1, 4096), torch.float32)
        buf4 = reinterpret_tensor(buf2, (1, 4080, 1), (4096, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((1, 4080, 3072), (12533760, 3072, 1), torch.bfloat16)
        # Source Nodes: [add_1, add_2, img, l__self___img_norm2, mul, mul_1], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_0.run(buf4, primals_16, primals_15, buf0, primals_18, primals_19, buf1, buf5, 4080, 3072, grid=grid(4080), stream=stream0)
        del primals_19
        buf6 = empty_strided_cuda((4080, 12288), (12288, 1), torch.bfloat16)
        # Source Nodes: [l__self___img_mlp_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_4, reinterpret_tensor(buf5, (4080, 3072), (3072, 1), 0), reinterpret_tensor(primals_3, (3072, 12288), (1, 3072), 0), alpha=1, beta=1, out=buf6)
        del primals_4
        buf7 = empty_strided_cuda((1, 4080, 12288), (50135040, 12288, 1), torch.bfloat16)
        # Source Nodes: [l__self___img_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_1.run(buf6, buf7, 50135040, grid=grid(50135040), stream=stream0)
        buf8 = empty_strided_cuda((4080, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [l__self___img_mlp_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf7, (4080, 12288), (12288, 1), 0), reinterpret_tensor(primals_5, (12288, 3072), (1, 12288), 0), alpha=1, beta=1, out=buf8)
        del primals_6
        buf9 = empty_strided_cuda((1, 4080, 3072), (12533760, 3072, 1), torch.bfloat16)
        # Source Nodes: [img, img_1, mul, mul_2], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_2.run(primals_16, primals_15, buf0, primals_17, buf8, buf9, 12533760, grid=grid(12533760), stream=stream0)
        buf10 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [l__self___txt_attn_proj], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(primals_13, (256, 3072), (3072, 1), 0), reinterpret_tensor(primals_7, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf10)
        del primals_8
        buf11 = empty_strided_cuda((1, 256, 1), (256, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 256, 1), (256, 1, 256), torch.float32)
        buf14 = reinterpret_tensor(buf12, (1, 256, 1), (256, 1, 1), 0); del buf12  # reuse
        buf15 = empty_strided_cuda((1, 256, 3072), (786432, 3072, 1), torch.bfloat16)
        # Source Nodes: [add_5, add_6, l__self___txt_norm2, mul_3, mul_4, txt], Original ATen: [aten.add, aten.mul, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_3.run(buf14, primals_14, primals_20, buf10, primals_22, primals_23, buf11, buf15, 256, 3072, grid=grid(256), stream=stream0)
        del primals_23
        buf16 = empty_strided_cuda((256, 12288), (12288, 1), torch.bfloat16)
        # Source Nodes: [l__self___txt_mlp_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf15, (256, 3072), (3072, 1), 0), reinterpret_tensor(primals_9, (3072, 12288), (1, 3072), 0), alpha=1, beta=1, out=buf16)
        del primals_10
        buf17 = empty_strided_cuda((1, 256, 12288), (3145728, 12288, 1), torch.bfloat16)
        # Source Nodes: [l__self___txt_mlp_1], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_4.run(buf16, buf17, 3145728, grid=grid(3145728), stream=stream0)
        buf18 = empty_strided_cuda((256, 3072), (3072, 1), torch.bfloat16)
        # Source Nodes: [l__self___txt_mlp_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf17, (256, 12288), (12288, 1), 0), reinterpret_tensor(primals_11, (12288, 3072), (1, 12288), 0), alpha=1, beta=1, out=buf18)
        del primals_12
        buf19 = empty_strided_cuda((1, 256, 3072), (786432, 3072, 1), torch.bfloat16)
        # Source Nodes: [mul_3, mul_5, txt, txt_1], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_5.run(primals_14, primals_20, buf10, primals_21, buf18, buf19, 786432, grid=grid(786432), stream=stream0)
    return (buf9, buf19, primals_14, primals_15, primals_16, primals_17, primals_18, primals_20, primals_21, primals_22, reinterpret_tensor(primals_13, (4080, 3072), (3072, 1), 786432), buf0, buf1, buf4, reinterpret_tensor(buf5, (4080, 3072), (3072, 1), 0), buf6, reinterpret_tensor(buf7, (4080, 12288), (12288, 1), 0), buf8, reinterpret_tensor(primals_13, (256, 3072), (3072, 1), 0), buf10, buf11, buf14, reinterpret_tensor(buf15, (256, 3072), (3072, 1), 0), buf16, reinterpret_tensor(buf17, (256, 12288), (12288, 1), 0), buf18, reinterpret_tensor(primals_11, (3072, 12288), (12288, 1), 0), reinterpret_tensor(primals_9, (12288, 3072), (3072, 1), 0), reinterpret_tensor(primals_7, (3072, 3072), (3072, 1), 0), reinterpret_tensor(primals_5, (3072, 12288), (12288, 1), 0), reinterpret_tensor(primals_3, (12288, 3072), (3072, 1), 0), reinterpret_tensor(primals_1, (3072, 3072), (3072, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((1, 4336, 3072), (13320192, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((1, 256, 3072), (786432, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((1, 4080, 3072), (12533760, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_17 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_19 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_20 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_22 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_23 = rand_strided((1, 1, 3072), (3072, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
