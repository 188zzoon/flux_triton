
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.debug = True
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = 'TRITON'
torch._inductor.config.trace.enabled = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.1+cu121
# torch cuda version: 12.1
# torch git version: 38b96d3399a695e704ed39b60dac733c3fbf20e2


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_9, torch.float32);  primals_9 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type)
        mul = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = sigmoid = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm = torch.ops.aten.addmm.default(primals_2, convert_element_type_1, permute);  primals_2 = permute = None
        unsqueeze = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        split = torch.ops.aten.split.Tensor(unsqueeze, 3072, -1);  unsqueeze = None
        getitem = split[0]
        getitem_1 = split[1];  split = None
        permute_1 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_4, convert_element_type_1, permute_1);  primals_4 = permute_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        split_1 = torch.ops.aten.split.Tensor(unsqueeze_1, 3072, -1);  unsqueeze_1 = None
        getitem_6 = split_1[0]
        getitem_7 = split_1[1];  split_1 = None
        add = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(add, primals_10);  add = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, getitem);  mul_2 = getitem = None
        view = torch.ops.aten.view.default(add_1, [40800, 3072]);  add_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm_2 = torch.ops.aten.addmm.default(primals_6, view, permute_2);  primals_6 = None
        view_1 = torch.ops.aten.view.default(addmm_2, [10, 4080, 9216]);  addmm_2 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_7, 1);  getitem_7 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_2, primals_11);  add_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, getitem_6);  mul_3 = getitem_6 = None
        view_2 = torch.ops.aten.view.default(add_3, [2560, 3072]);  add_3 = None
        permute_3 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_3 = torch.ops.aten.addmm.default(primals_8, view_2, permute_3);  primals_8 = None
        view_3 = torch.ops.aten.view.default(addmm_3, [10, 256, 9216]);  addmm_3 = None
        view_4 = torch.ops.aten.view.default(view_1, [10, 4080, 3, 24, 128]);  view_1 = None
        permute_4 = torch.ops.aten.permute.default(view_4, [2, 0, 3, 1, 4]);  view_4 = None
        select = torch.ops.aten.select.int(permute_4, 0, 0)
        select_1 = torch.ops.aten.select.int(permute_4, 0, 1)
        select_2 = torch.ops.aten.select.int(permute_4, 0, 2);  permute_4 = None
        view_5 = torch.ops.aten.view.default(view_3, [10, 256, 3, 24, 128]);  view_3 = None
        permute_5 = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
        select_3 = torch.ops.aten.select.int(permute_5, 0, 0)
        select_4 = torch.ops.aten.select.int(permute_5, 0, 1)
        select_5 = torch.ops.aten.select.int(permute_5, 0, 2);  permute_5 = None
        cat = torch.ops.aten.cat.default([select_3, select], 2);  select_3 = select = None
        cat_1 = torch.ops.aten.cat.default([select_4, select_1], 2);  select_4 = select_1 = None
        cat_2 = torch.ops.aten.cat.default([select_5, select_2], 2);  select_5 = select_2 = None
        permute_8 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_12 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return [cat, cat_1, cat_2, primals_10, primals_11, convert_element_type_1, view, view_2, permute_8, permute_12]
        
def load_args(reader):
    buf0 = reader.storage(None, 113246208, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (18432, 3072), dtype=torch.float16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (18432,), dtype=torch.float16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 113246208, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (18432, 3072), dtype=torch.float16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf3, (18432,), dtype=torch.float16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 56623104, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf4, (9216, 3072), dtype=torch.float16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 18432, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf5, (9216,), dtype=torch.float16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 56623104, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf6, (9216, 3072), dtype=torch.float16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 18432, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf7, (9216,), dtype=torch.float16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 61440, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf8, (10, 3072), dtype=torch.float16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 250675200, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf9, (10, 4080, 3072), dtype=torch.float16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 15728640, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf10, (10, 256, 3072), dtype=torch.float16, is_leaf=True)  # primals_11
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)