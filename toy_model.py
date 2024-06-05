from net import *
import torch.nn.functional as F

config = {
    'horizon': 100,
    'state_dim': 1,
    'action_dim': 10,
    'n_layer': 1,
    'n_embd': 64,
    'n_head': 8,
    'dropout': False,
    'lr': 0.001,
    'dim': 10,
    'seed': 1,
    'test': False,
    'store_gpu': True,
    'n_envs': 100
}

model = Transformer(config)

print(model.transformer)

head_0 = model.transformer.h[0].attn
print("c_attn.weight.shape:", head_0.c_attn.weight.shape)


# head_1 = model.transformer.h[1].attn
# K_1 = head_1.c_attn.weight[:, :16]

# head_2 = model.transformer.h[2].attn
# K_2 = head_2.c_attn.weight[:, :16]

# head_3 = model.transformer.h[3].attn
# K_3 = head_3.c_attn.weight[:, :16]

# # Find cosine similarity between the keys of the first head and the keys of the other heads
# cos_sim_0_1 = F.cosine_similarity(K_0, K_1, dim=1)
# cos_sim_0_2 = F.cosine_similarity(K_0, K_2, dim=1)
# cos_sim_0_3 = F.cosine_similarity(K_0, K_3, dim=1)

# # Print the cosine similarities
# print(cos_sim_0_1)
# print(cos_sim_0_2)
# print(cos_sim_0_3)
