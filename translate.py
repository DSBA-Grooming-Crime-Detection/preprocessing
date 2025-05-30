import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "nayohan/llama3-instrucTrans-enko-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map="auto",
  torch_dtype=torch.bfloat16
)

system_prompt="당신은 번역기 입니다. 영어를 한국어로 번역하세요."
sentence = ["Hey, you seem really cool. Do you wanna be friends?",
"Don’t worry, you can tell me anything. I won’t judge.",
"Your parents don’t get you like I do.",
"You’re so mature for your age.",
"You don’t need to tell anyone about our chats. They just wouldn’t understand.",
"This is our little secret, okay?",
"If anyone finds out, we could both get in trouble.",
"I trust you not to show this to anyone.",
"Can you send me a cute picture of you? Just for me.",
"If you really care about me, you’ll prove it.",
"Don’t be shy, it’s just a picture.",
"It’s totally normal, I’ve done this with other friends too.",
"Let’s meet, but don’t tell your parents. It’ll be our little adventure.",
"You’re so special to me, I wish we could be alone together.",
"I can send you something too… but you first."
]

conversation = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': sentence}]

inputs = tokenizer.apply_chat_template(
  conversation,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors='pt'
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=4096) # Finetuned with length 4096
print(tokenizer.decode(outputs[0][len(inputs[0]):]))
