import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "In one sentence, what is a GPU?"


def run_inference(model, tokenizer, device):
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    return elapsed, text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    # Warmup inference so CUDA kernels/context are fully initialized before
    # we report readiness -- this is the moment cuda-checkpoint cares about.
    warmup_elapsed, _ = run_inference(model, tokenizer, device)
    print(f"READY pid={__import__('os').getpid()} warmup_s={warmup_elapsed:.3f}", flush=True)

    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "infer":
            elapsed, text = run_inference(model, tokenizer, device)
            print(f"INFER_DONE elapsed_s={elapsed:.3f} text={text!r}", flush=True)
        elif cmd == "exit":
            break


if __name__ == "__main__":
    main()
