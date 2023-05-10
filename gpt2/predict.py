import os
from transformers import AutoTokenizer,GPT2LMHeadModel

model_name = 'model_output/checkpoint-10'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name,pad_token_id=tokenizer.eos_token_id)

while True:
    inp = input('请输入：')
    input_ids = tokenizer.encode(inp,return_tensors='pt')

    beam_output = model.generate(
        input_ids,
        max_length=500,
        num_beams=4,
        no_repeat_ngram_size=1,
        length_penalty=1.34,
        early_stopping=True
    )

    output_text=  tokenizer.decode(output[0],skip_special_tokens=True)

    print('输出:',output_text)
