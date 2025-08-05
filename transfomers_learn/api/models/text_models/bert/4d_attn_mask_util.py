import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# 假设 batch_size=2, seq_len=5
attention_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0]
], dtype=torch.float32)

print("输入 attention_mask:")
print(attention_mask)

# 非causal 4d mask
noncausal_converter = AttentionMaskConverter(is_causal=False)
noncausal_4d = noncausal_converter.to_4d(attention_mask, query_length=3, dtype=torch.float32, key_value_length=5)
print("\n非causal 4D mask:")
print(noncausal_4d)
print("shape:", noncausal_4d.shape)

# causal 4d mask
causal_converter = AttentionMaskConverter(is_causal=True)
causal_4d = causal_converter.to_4d(attention_mask, query_length=3, dtype=torch.float32, key_value_length=5)
print("\ncausal 4D mask:")
print(causal_4d)
print("shape:", causal_4d.shape)

# 输入 attention_mask:
# tensor([[1., 1., 1., 0., 0.],
#         [1., 1., 0., 0., 0.]])

# 非causal 4D mask:
# tensor([[[[ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38]]],


#         [[[ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38]]]])
# shape: torch.Size([2, 1, 3, 5])

# causal 4D mask:
# tensor([[[[ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38]]],


#         [[[ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38],
#           [ 0.0000e+00,  0.0000e+00, -3.4028e+38, -3.4028e+38, -3.4028e+38]]]])