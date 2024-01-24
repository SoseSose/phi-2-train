from transformers import AutoTokenizer, AutoModelForCausalLM   

MAX_TOKENS = 2048

tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto",
    # device_map="cuda:0",
    device_map="auto",
    cache_dir ="D:/models",
    trust_remote_code=True,
)
prompt = "My name is"
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
output_ids = model.generate(
    token_ids.to(model.device),
    temperature=0.2,
    do_sample=True,
    max_length=MAX_TOKENS,
)
answer = tokenizer.decode(output_ids[0][token_ids.size(1) :])