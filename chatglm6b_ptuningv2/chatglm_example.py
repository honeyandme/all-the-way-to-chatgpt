from transformers import AutoTokenizer, AutoModel
#download from https://huggingface.co/THUDM/chatglm-6b-int4
tokenizer = AutoTokenizer.from_pretrained("chat_glm_int4", trust_remote_code=True)
model = AutoModel.from_pretrained("chat_glm_int4", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
