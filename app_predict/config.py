import torch
import os

from SimilarityServer.settings import BASE_DIR



# model_path
check_point = "voidful/albert_chinese_tiny"
pretrain_model_path = os.path.join(BASE_DIR, "app_predict/local_model")
pretrain_tokenizer_path = os.path.join(BASE_DIR, "app_predict/local_model")
model_type = "albert_chinese_tiny"

transformers_instant_type = "Auto"

pretrain_hidden_size = 312 if model_type == "albert_chinese_tiny" else 768

# 在 django 中 设置文件路径要跟 basedir 联动
best_model_path = os.path.join(BASE_DIR, "app_predict/best_model/best_model_test.pk1")


# tokenizer
token_padding = True
token_truncation = True
# 训练推荐参数
token_max_length = 64
return_tensors = 'pt'

# pytorch
# dataloader
# 训练推荐参数
batch_size = 64
# batch_size = 128
dataloader_shuffle = True
dataloader_dropout = False

# cross_attention
cross_attention_n_heads = 4

# 这里跟pretrain_model的输出维度保持一致
# 这里会变成浮点数，要变成整数
cross_attention_d_q = round(pretrain_hidden_size/4)
cross_attention_d_k = round(pretrain_hidden_size/4)
cross_attention_d_v = round(pretrain_hidden_size/4)

# similarity
similarity_drop_out = 0.1
num_labels = 2

# cuda
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# similarity
similarity_window_size = 7
