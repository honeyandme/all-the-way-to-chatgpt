import os
from glob import glob
import random
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizerFast,GPT2Config,GPT2LMHeadModel,DataCollatorForLanguageModeling
from transformers import Trainer,TrainingArguments
import wandb
os.environ["WANDB_DISABLED"] = "true"
def tokenize(element):
    outputs = tokenizer(element['content'],truncation=True,max_length=max_input_len,return_overflowing_tokens=True,return_length=True)
    batch_input = []
    for input_id,input_len in zip(outputs['input_ids'],outputs['length']):
        if input_len == max_input_len:
            batch_input.append(input_id)

    return {"input_ids":batch_input}
if __name__ == "__main__":
    random.seed(3407)
    file_path = glob(pathname=os.path.join('gpt2_data_mini','*','*'))

    test_rate = 0.15
    max_input_len = 128
    test_data = random.sample(file_path,int(len(file_path)*test_rate))
    train_data = [i for i in file_path if i not in test_data]

    raw_dataset = load_dataset("csv",data_files={"train":train_data,"test":test_data},cache_dir="cache_dir")

    tokenizer = BertTokenizerFast.from_pretrained(os.path.join('..','..','nlp','data','bert_base_chinese'))
    tokenizer.add_special_tokens({"bos_token":"[begin]","eos_token":"[end]",'pad_token': '[PAD]'})
    tokenized_dataset = raw_dataset.map(tokenize,batched=True,remove_columns=raw_dataset['train'].column_names)

    config = GPT2Config.from_pretrained("gpt2",
                                        vocab_size = len(tokenizer),
                                        n_ctx = max_input_len,
                                        bos_token_id = tokenizer.bos_token_id,
                                        eos_token_id = tokenizer.eos_token_id
                                        )

    model = GPT2LMHeadModel(config).to('mps')

    data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)
    args = TrainingArguments(
        output_dir="model_output",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=2000,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=10,
        # fp16=True,
        push_to_hub=False,
    )

    trianer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trianer.train()