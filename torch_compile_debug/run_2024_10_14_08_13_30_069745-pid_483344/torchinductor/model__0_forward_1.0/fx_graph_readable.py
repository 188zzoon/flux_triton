class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f16[18432, 3072]", primals_2: "f16[18432]", primals_3: "f16[18432, 3072]", primals_4: "f16[18432]", primals_5: "f16[9216, 3072]", primals_6: "f16[9216]", primals_7: "f16[9216, 3072]", primals_8: "f16[9216]", primals_9: "f16[10, 3072]", primals_10: "f16[10, 4080, 3072]", primals_11: "f16[10, 256, 3072]"):
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:154 in forward, code: out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        convert_element_type: "f32[10, 3072]" = torch.ops.prims.convert_element_type.default(primals_9, torch.float32);  primals_9 = None
        sigmoid: "f32[10, 3072]" = torch.ops.aten.sigmoid.default(convert_element_type)
        mul: "f32[10, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = sigmoid = None
        convert_element_type_1: "f16[10, 3072]" = torch.ops.prims.convert_element_type.default(mul, torch.float16);  mul = None
        permute: "f16[3072, 18432]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f16[10, 18432]" = torch.ops.aten.addmm.default(primals_2, convert_element_type_1, permute);  primals_2 = permute = None
        unsqueeze: "f16[10, 1, 18432]" = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        split = torch.ops.aten.split.Tensor(unsqueeze, 3072, -1);  unsqueeze = None
        getitem: "f16[10, 1, 3072]" = split[0]
        getitem_1: "f16[10, 1, 3072]" = split[1];  split = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:154 in forward, code: out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        permute_1: "f16[3072, 18432]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        addmm_1: "f16[10, 18432]" = torch.ops.aten.addmm.default(primals_4, convert_element_type_1, permute_1);  primals_4 = permute_1 = None
        unsqueeze_1: "f16[10, 1, 18432]" = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        split_1 = torch.ops.aten.split.Tensor(unsqueeze_1, 3072, -1);  unsqueeze_1 = None
        getitem_6: "f16[10, 1, 3072]" = split_1[0]
        getitem_7: "f16[10, 1, 3072]" = split_1[1];  split_1 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:219 in forward, code: img_modulated = (1 + img_mod1.scale) * img + img_mod1.shift
        add: "f16[10, 1, 3072]" = torch.ops.aten.add.Tensor(getitem_1, 1);  getitem_1 = None
        mul_2: "f16[10, 4080, 3072]" = torch.ops.aten.mul.Tensor(add, primals_10);  add = None
        add_1: "f16[10, 4080, 3072]" = torch.ops.aten.add.Tensor(mul_2, getitem);  mul_2 = getitem = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:220 in forward, code: img_qkv = self.img_attn.qkv(img_modulated)
        view: "f16[40800, 3072]" = torch.ops.aten.view.default(add_1, [40800, 3072]);  add_1 = None
        permute_2: "f16[3072, 9216]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm_2: "f16[40800, 9216]" = torch.ops.aten.addmm.default(primals_6, view, permute_2);  primals_6 = None
        view_1: "f16[10, 4080, 9216]" = torch.ops.aten.view.default(addmm_2, [10, 4080, 9216]);  addmm_2 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:234 in forward, code: txt_modulated = (1 + txt_mod1.scale) * txt + txt_mod1.shift
        add_2: "f16[10, 1, 3072]" = torch.ops.aten.add.Tensor(getitem_7, 1);  getitem_7 = None
        mul_3: "f16[10, 256, 3072]" = torch.ops.aten.mul.Tensor(add_2, primals_11);  add_2 = None
        add_3: "f16[10, 256, 3072]" = torch.ops.aten.add.Tensor(mul_3, getitem_6);  mul_3 = getitem_6 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:235 in forward, code: txt_qkv = self.txt_attn.qkv(txt_modulated)
        view_2: "f16[2560, 3072]" = torch.ops.aten.view.default(add_3, [2560, 3072]);  add_3 = None
        permute_3: "f16[3072, 9216]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_3: "f16[2560, 9216]" = torch.ops.aten.addmm.default(primals_8, view_2, permute_3);  primals_8 = None
        view_3: "f16[10, 256, 9216]" = torch.ops.aten.view.default(addmm_3, [10, 256, 9216]);  addmm_3 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:239 in forward, code: img_q, img_k, img_v = rearrange(
        view_4: "f16[10, 4080, 3, 24, 128]" = torch.ops.aten.view.default(view_1, [10, 4080, 3, 24, 128]);  view_1 = None
        permute_4: "f16[3, 10, 24, 4080, 128]" = torch.ops.aten.permute.default(view_4, [2, 0, 3, 1, 4]);  view_4 = None
        select: "f16[10, 24, 4080, 128]" = torch.ops.aten.select.int(permute_4, 0, 0)
        select_1: "f16[10, 24, 4080, 128]" = torch.ops.aten.select.int(permute_4, 0, 1)
        select_2: "f16[10, 24, 4080, 128]" = torch.ops.aten.select.int(permute_4, 0, 2);  permute_4 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:243 in forward, code: txt_q, txt_k, txt_v = rearrange(
        view_5: "f16[10, 256, 3, 24, 128]" = torch.ops.aten.view.default(view_3, [10, 256, 3, 24, 128]);  view_3 = None
        permute_5: "f16[3, 10, 24, 256, 128]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
        select_3: "f16[10, 24, 256, 128]" = torch.ops.aten.select.int(permute_5, 0, 0)
        select_4: "f16[10, 24, 256, 128]" = torch.ops.aten.select.int(permute_5, 0, 1)
        select_5: "f16[10, 24, 256, 128]" = torch.ops.aten.select.int(permute_5, 0, 2);  permute_5 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:248 in forward, code: q = torch.cat((txt_q, img_q), dim=2)
        cat: "f16[10, 24, 4336, 128]" = torch.ops.aten.cat.default([select_3, select], 2);  select_3 = select = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:249 in forward, code: k = torch.cat((txt_k, img_k), dim=2)
        cat_1: "f16[10, 24, 4336, 128]" = torch.ops.aten.cat.default([select_4, select_1], 2);  select_4 = select_1 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:250 in forward, code: v = torch.cat((txt_v, img_v), dim=2)
        cat_2: "f16[10, 24, 4336, 128]" = torch.ops.aten.cat.default([select_5, select_2], 2);  select_5 = select_2 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:235 in forward, code: txt_qkv = self.txt_attn.qkv(txt_modulated)
        permute_8: "f16[9216, 3072]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
        # File: /home/myeongjun/workspace/flux_triton/src/flux/modules/layers_qkv.py:220 in forward, code: img_qkv = self.img_attn.qkv(img_modulated)
        permute_12: "f16[9216, 3072]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return [cat, cat_1, cat_2, primals_10, primals_11, convert_element_type_1, view, view_2, permute_8, permute_12]
        