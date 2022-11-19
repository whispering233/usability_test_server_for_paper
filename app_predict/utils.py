from transformers import AutoTokenizer, AutoModel, BertModel, AlbertForMaskedLM
import app_predict.config as config


# transformers
# 实例化tokenizer
def instant_tokenizer(check_point=config.pretrain_tokenizer_path):
    return AutoTokenizer.from_pretrained(check_point)


tokenizers = instant_tokenizer()


def tokenize_function(text):
    # 这里有参数 padding='max_length'
    # 与 padding=True 之间的区别
    inputs =  tokenizers(text,
                         padding='max_length',
                         truncation=config.token_truncation,
                         max_length=config.token_max_length,
                         return_tensors=config.return_tensors)

    for _ in inputs:
        inputs[_] = inputs[_].to(config.device)
    return inputs


# 返回指定的预处理模型
def instant_pretrain_model(check_point=config.pretrain_model_path, trans_instant_type=config.transformers_instant_type):
    if trans_instant_type == "Auto":
        # return AutoModel.from_pretrained(config.check_point)
        return AutoModel.from_pretrained(check_point)
    else:
        return BertModel.from_pretrained(check_point)


# 实例化指定的预处理模型
pretrain_model = instant_pretrain_model()
