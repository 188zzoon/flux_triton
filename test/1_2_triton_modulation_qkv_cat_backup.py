import torch
import torch._inductor.config
# torch._inductor.config.trace.enabled = True
# torch._inductor.config.max_autotune_gemm_backends = "TRITON"
# torch._inductor.config.max_autotune = True
# torch._inductor.config.debug = True
from flux.modules.layers_qkv import DoubleStreamBlock

import triton
import triton.language as tl
import torch

torch_device = torch.device("cuda")
torch_dtype = torch.float16

from torch._inductor.runtime.triton_heuristics import grid

def mm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), 1, 1)

meta1 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}

@triton.jit
def _triton_silu(x_ptr, b_ptr, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x = tl.load(x_ptr + x0, mask=xmask).to(tl.float32)
    output = (x * tl.sigmoid(x)).to(tl.float32)
    tl.store(b_ptr + x0, output, xmask)


def triton_silu(x: torch.Tensor):
    output = torch.empty_like(x)
    xnumel = x.numel()
    grid = lambda META: (META["XBLOCK"],)
    _triton_silu[grid](x, output, xnumel, XBLOCK=4096)
    return output

@triton.jit
def _triton_linear(a_ptr, b_ptr, c_ptr, out_ptr,
                    M, N, K,
                    stride_am, stride_ak, 
                    stride_bk, stride_bn,
                    GROUP_M : tl.constexpr,
                    EVEN_K : tl.constexpr,
                    ALLOW_TF32 : tl.constexpr,
                    ACC_TYPE : tl.constexpr,
                    B_PROLOGUE_CAST_TYPE : tl.constexpr,
                    BLOCK_M : tl.constexpr,
                    BLOCK_N : tl.constexpr,
                    BLOCK_K : tl.constexpr,
                    ):

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offset_am = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    else:
        offset_am = offset_m % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offset_bn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    else:
        offset_bn = offset_n % N
    offset_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offset_k[None, :] < k, other=0.)
            b = tl.load(b_ptrs, mask=offset_k[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # rematerialize offset_m and offset_n to save registers
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = offset_m[:, None]
    idx_n = offset_n[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (N*idx_m)
    c = tl.load(c_ptr + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    out = acc + c
    tl.store(out_ptr + (tl.broadcast_to(xindex, mask.shape)), out, mask)

def triton_linear(x, linear):
    
    weight = linear.weight.T
    bias = linear.bias

    M, K = x.shape
    K, N = weight.shape

    stride_am, stride_ak = x.stride()
    stride_wk, stride_wn = weight.stride()

    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = (288, 1, 1)
    _triton_linear[grid](
            x, weight, bias, output,
            M, N, K, 
            stride_am, stride_ak, 
            stride_wk, stride_wn,
            GROUP_M= 8,
            EVEN_K=True,
            ALLOW_TF32=False,
            ACC_TYPE=tl.float32,
            B_PROLOGUE_CAST_TYPE=None,
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=64,
        )

    return output

def triton_linear_matrix(x, linear):
    
    weight = linear.weight.T
    bias = linear.bias

    M = x.shape[0] * x.shape[1]
    K, N = weight.shape

    stride_am, stride_ak = x.stride(1), x.stride(2)
    stride_wk, stride_wn = weight.stride()

    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = (64, 72, 1)
    _triton_linear[grid](
            x, weight, bias, output,
            M, N, K, 
            stride_am, stride_ak, 
            stride_wk, stride_wn,
            GROUP_M= 8,
            EVEN_K=True,
            ALLOW_TF32=False,
            ACC_TYPE=tl.float32,
            B_PROLOGUE_CAST_TYPE=None,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )

    return output


def silu_linear_forward(vec, lin):
    buf0 = triton_silu(vec)
    output = triton_linear(buf0, lin)
    return output

def qkv_linear(img, linear):
    weight = linear.weight.T
    bias = linear.bias

    x = img.view(-1, weight.shape[0])

    M, K = x.shape
    _, N = weight.shape

    output = torch.empty((M, N), dtype=img.dtype, device=img.device)

    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = weight.stride()

    grid = mm_grid(M, N, meta1)
    _triton_linear[grid](
            x, weight, bias, output,
            M, N, K, 
            stride_am, stride_ak, 
            stride_bk, stride_bn,
            GROUP_M= 8,
            EVEN_K=True,
            ALLOW_TF32=False,
            ACC_TYPE=tl.float32,
            B_PROLOGUE_CAST_TYPE=None,
            BLOCK_M=128,
            BLOCK_N=128,
            BLOCK_K=32,
        )

    return output.view(img.shape[0], img.shape[1], weight.shape[1])


@triton.jit
def mod1_modulation(img_ptr, modulation_ptr, output_ptr, batch_size, head_size, modulation_size, XBLOCK : tl.constexpr):
    
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK + tl.arange(0, XBLOCK)[:]

    batch_idx = (xoffset // batch_size) # x2
    head_dim_idx = xoffset % head_size  # x0

    modulation_offset = head_dim_idx + (modulation_size * batch_idx)

    img = tl.load(img_ptr + (xoffset), None)
    shift = tl.load(modulation_ptr + (modulation_offset + head_size * 0), None, eviction_policy='evict_last')
    scale = tl.load(modulation_ptr + (modulation_offset + head_size * 1), None, eviction_policy='evict_last')

    output = (scale + 1.0) * img + shift
    tl.store(output_ptr + (xoffset), output, None)

def mod1_modulation_forward(img, mod):
    # compute
    # img_modulated = (1 + img_mod1.scale) * img + img_mod1.shift
    output = torch.empty_like(img)
    batch_size = img.stride()[0]  # 12533760 = 4080 * 3072 = seq_len * num_head * head_dim
    head_size = img.stride()[1] # 3072  # num_head * head_dim
    modulation_size = mod.stride()[0]   # head_dim * 6 = 18432
    xnumel = img.numel() # 10 * 4080 * 3072
    _grid = grid(xnumel)
    mod1_modulation[_grid](img, mod, output, batch_size, head_size, modulation_size, XBLOCK=1024)
    return output


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

@triton.jit
def triton_qkv_concat(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x2 = (xindex // 3072) % 4336
    x3 = (xindex // 13320192)
    x4 = xindex % 3072
    x0 = xindex % 128
    x1 = (xindex // 128) % 24
    tmp0 = x2
    
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    
    tmp11 = tl.load(in_ptr1 + ((-2359296) + x4 + (9216*x2) + (37601280*x3)), tmp8, other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14, None)

##############################################################################################################
##############################################################################################################
    tmp5 = tl.load(in_ptr0 + (3072 + x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
   
    tmp11 = tl.load(in_ptr1 + ((-2356224) + x4 + (9216*x2) + (37601280*x3)), tmp8, other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr1 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14, None)

##############################################################################################################
##############################################################################################################
    tmp5 = tl.load(in_ptr0 + (6144 + x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
   
    tmp11 = tl.load(in_ptr1 + ((-2353152) + x4 + (9216*x2) + (37601280*x3)), tmp8, other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr2 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14, None)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# @triton.jit
# def triton_qkv_concat():

@triton.jit
def triton_fail(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26640384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3072) % 4336
    x3 = (xindex // 13320192)
    x4 = xindex % 3072
    x0 = xindex % 128
    x1 = (xindex // 128) % 24
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    ####################################################################################################
    #TODO index 분리
    #qvk
    # n * 3072
    # q = 0 * 3072
    # k = 1 * 3072
    # v = 2 * 3072
    tmp5 = tl.load(in_ptr0 + (x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    tmp5_k = tl.load(in_ptr0 + (3072 + x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    tmp5_v = tl.load(in_ptr0 + (6144 + x4 + (9216*x2) + (2359296*x3)), tmp4, other=0.0).to(tl.float32)
    ####################################################################################################
    #q
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 4336, tl.int64)
    tmp10 = tmp0 < tmp9

    #k
    tmp6_k = tl.full(tmp5_k.shape, 0.0, tmp5_k.dtype)
    tmp7_k = tl.where(tmp4, tmp5_k, tmp6_k)
    tmp8_k = tmp0 >= tmp3
    tmp9_k = tl.full([1], 4336, tl.int64)
    tmp10_k = tmp0 < tmp9_k

    #v
    tmp6_v = tl.full(tmp5_v.shape, 0.0, tmp5_v.dtype)
    tmp7_v = tl.where(tmp4, tmp5_v, tmp6_v)
    tmp8_v = tmp0 >= tmp3
    tmp9_v = tl.full([1], 4336, tl.int64)
    tmp10_v = tmp0 < tmp9_v

    ####################################################################################################
    #TODO index 분리 -2359296 -2356224 -2353152 => 3072 간격
    # q = (-2359296) + (3072 * 0)
    # k = (-2359296) + (3072 * 1)
    # v = (-2359296) + (3072 * 2)
    tmp11 = tl.load(in_ptr1 + ((-2359296) + x4 + (9216*x2) + (37601280*x3)), tmp8, other=0.0).to(tl.float32)
    tmp11_k = tl.load(in_ptr1 + ((-2356224) + x4 + (9216*x2) + (37601280*x3)), tmp8_k, other=0.0).to(tl.float32)
    tmp11_v = tl.load(in_ptr1 + ((-2353152) + x4 + (9216*x2) + (37601280*x3)), tmp8_v, other=0.0).to(tl.float32)
    ####################################################################################################
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)

    tmp12_k = tl.full(tmp11_k.shape, 0.0, tmp11_k.dtype)
    tmp13_k = tl.where(tmp8_k, tmp11_k, tmp12_k)
    tmp14_k = tl.where(tmp4, tmp7_k, tmp13_k)

    tmp12_v = tl.full(tmp11_v.shape, 0.0, tmp11_v.dtype)
    tmp13_v = tl.where(tmp8_v, tmp11_k, tmp12_v)
    tmp14_v = tl.where(tmp4, tmp7_v, tmp13_v)
    ####################################################################################################
    #TODO index 분리
    # output_q, output_k, output_v
    tl.store(out_ptr0 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14, None)
    tl.store(out_ptr1 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14_k, None)
    tl.store(out_ptr2 + (x0 + (128*x2) + (555008*x1) + (13320192*x3)), tmp14_v, None)
    ####################################################################################################
    
    return output_q, output_k, output_v

def qkv_concat(img_qkv, txt_qkv, num_heads):
    # shape : (batch_size, num_heads, seq_len1+seq_len2, head_dims)
    # 3072 = num_heads * head_dims
    # => head_dims = 128
    # 24 * 128 = 3072
    # 4336 = 4080 + 256 (seq_len)
    # buf9 = empty_strided_cuda((10, 24, 4336, 128), (13320192, 555008, 128, 1), torch.float16)
    # img_qkv.shape : 10, 4080, 9216
    # txt_qkv.shape : 10, 256, 9216
    batch_size, seq_img, hidden_dim= img_qkv.shape
    _, seq_txt, _ = txt_qkv.shape
    head_dim = hidden_dim // (3 * num_heads)
    seq = seq_img + seq_txt
    output_q = torch.empty((10, 24, 4336, 128), dtype=img_qkv.dtype, device=img_qkv.device)
    output_k = torch.empty((10, 24, 4336, 128), dtype=img_qkv.dtype, device=img_qkv.device)
    output_v = torch.empty((10, 24, 4336, 128), dtype=img_qkv.dtype, device=img_qkv.device)

    output_q = torch.empty((batch_size, num_heads, seq, head_dim), dtype=img_qkv.dtype, device=img_qkv.device)
    output_k = torch.empty((batch_size, num_heads, seq, head_dim), dtype=img_qkv.dtype, device=img_qkv.device)
    output_v = torch.empty((batch_size, num_heads, seq, head_dim), dtype=img_qkv.dtype, device=img_qkv.device)

    # grid = (26640384,)
    # grid = (6504, 1, 1)
    # grid = (130080,1,1)
    # grid = lambda META: (META["XBLOCK"],)
    grid = (130080,1,1)
    triton_qkv_concat[grid](txt_qkv, img_qkv, output_q, output_k, output_v, 133201920, XBLOCK=1024)
    
    return output_q, output_k, output_v

if __name__ == "__main__":

    hidden_size=3072
    num_heads=24
    mlp_ratio=4.0
    qkv_bias=True
    
    # params
    # cos, sin = get_cos_sin() #shape : 1, 4336, 64
    batch_size = 10
    cos = torch.randn((batch_size,4336,64), device=torch_device, dtype=torch_dtype)
    sin = torch.randn((batch_size,4336,64), device=torch_device, dtype=torch_dtype)
    
    img = torch.randn((batch_size,4080,3072), device=torch_device, dtype=torch_dtype)

    txt = torch.randn((batch_size,256,3072), device=torch_device, dtype=torch_dtype)
    vec = torch.randn((batch_size,3072), device=torch_device, dtype=torch_dtype)
    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias).to(device=torch_device, dtype=torch_dtype)

    
    # output = block.img_mod(vec)
    
    # print()
    # fn = torch.compile(block)
    # output = fn(img, txt, vec, cos, sin)
    # output = block(img, txt, vec, cos, sin)
    # print(output.shape)

    # img_modulated, img_qkv = block(img, txt, vec, cos, sin)
    # txt_modulated, txt_qkv = block(img, txt, vec, cos, sin)
    # fn = torch.compile(block)
    q,k,v, gt_img_qkv, gt_txt_qkv = block(img, txt, vec, cos, sin)


    # q, k, v = fn(img, txt, vec, cos, sin)


    # # TODO img_modulated = self.img_norm1(img)
    #        txt_modulated = self.txt_norm1(txt)

    #### IMAGE
    mod = silu_linear_forward(vec, block.img_mod.lin)
    buf0_img = mod1_modulation_forward(img, mod)
    img_qkv = qkv_linear(buf0_img, block.img_attn.qkv)


    #### TEXT
    mod = silu_linear_forward(vec, block.txt_mod.lin)
    buf0_txt = mod1_modulation_forward(txt, mod)
    txt_qkv = qkv_linear(buf0_txt, block.txt_attn.qkv)

    output_q, output_k, output_v = qkv_concat(img_qkv, txt_qkv, num_heads)

    print()

    torch.testing.assert_close(gt_img_qkv, img_qkv, atol=1e-2, rtol=0)
    torch.testing.assert_close(gt_txt_qkv, txt_qkv, atol=1e-2, rtol=0)
    torch.testing.assert_close(q, output_q, atol=1e-2, rtol=0)
    torch.testing.assert_close(k, output_k, atol=1e-2, rtol=0)
    torch.testing.assert_close(v, output_v, atol=1e-2, rtol=0)

    print("DONE-9-buf20-qkv")
    print("DONE-9-buf20-qkv")