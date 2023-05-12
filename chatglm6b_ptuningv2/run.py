from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("glm_model", trust_remote_code=True)#download from https://huggingface.co/THUDM/chatglm-6b-int4
model = AutoModel.from_pretrained("glm_model", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
