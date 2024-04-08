from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = snapshot_download("Shanghai_AI_Laboratory/internlm-7b", revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
if __name__ == '__main__':

    # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and might cause OOM Error.
    model = AutoModelForCausalLM.from_pretrained(model_dir,device_map="auto", offload_folder="offload", trust_remote_code=True, torch_dtype=torch.float16)
    model = model.eval()
    inputs = tokenizer(["A beautiful flower"], return_tensors="pt")
    # for k,v in inputs.items():
    #     inputs[k] = v.cuda()
    gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.1}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)
    #A beautiful flower box made of white rose wood. It is a perfect gift for weddings, birthdays and anniversaries.
    #All the roses are from our farm Roses Flanders. Therefor you know that these flowers last much longer than those in store or online!</s>