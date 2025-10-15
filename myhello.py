from transformers import AutoConfig, AutoModel

model_path = "/home/promote/.cache/modelscope/hub/models/stepfun-ai/Step-Audio-2-min"

# 尝试加载配置
config = AutoConfig.from_pretrained(model_path)
print("Config loaded successfully:", config)

# 尝试加载模型（如果内存足够）
# model = AutoModel.from_pretrained(model_path)
# print("Model loaded successfully")