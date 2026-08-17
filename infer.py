import argparse
import os
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-loop",
        action="store_true",
        help="Run inference repeatedly on a timer instead of reading commands "
        "from stdin. Used for the container/CRIU test, where a stdin pipe "
        "isn't reliable across a process dump+restore.",
    )
    parser.add_argument("--loop-interval", type=float, default=5.0)
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write status lines to this file instead of stdout. Used in the "
        "container/CRIU test so output doesn't depend on a stdio pipe "
        "surviving a process dump+restore.",
    )
    parser.add_argument(
        "--fd-test-file",
        default=None,
        help="If set, open a handful of extra file descriptors (a raw os-level "
        "fd, a buffered Python file object, and a self-connected pipe) and "
        "exercise them every loop iteration, to check that CRIU dump/restore "
        "preserves open fds and their positions correctly, not just CUDA "
        "state. Counters are written to <path>.raw and <path>.pyfile.",
    )
    args = parser.parse_args()

    out = open(args.log_file, "a", buffering=1) if args.log_file else sys.stdout

    def log(msg):
        print(msg, file=out, flush=True)

    raw_fd = py_file = pipe_r = pipe_w = None
    if args.fd_test_file:
        raw_fd = os.open(args.fd_test_file + ".raw", os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
        py_file = open(args.fd_test_file + ".pyfile", "a+", buffering=1)
        pipe_r, pipe_w = os.pipe()

    def exercise_fds(counter):
        if raw_fd is None:
            return
        os.write(raw_fd, f"raw {counter}\n".encode())
        print(f"pyfile {counter}", file=py_file, flush=True)
        os.write(pipe_w, f"pipe {counter}\n".encode())
        readback = os.read(pipe_r, 256).decode().strip()
        log(f"FD_CHECK counter={counter} raw_fd={raw_fd} pipe_readback={readback!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    # Warmup inference so CUDA kernels/context are fully initialized before
    # we report readiness -- this is the moment cuda-checkpoint cares about.
    warmup_elapsed, _ = run_inference(model, tokenizer, device)
    log(f"READY pid={os.getpid()} warmup_s={warmup_elapsed:.3f}")
    exercise_fds(0)

    if args.auto_loop:
        counter = 0
        while True:
            elapsed, text = run_inference(model, tokenizer, device)
            log(f"INFER_DONE elapsed_s={elapsed:.3f} text={text!r}")
            counter += 1
            exercise_fds(counter)
            time.sleep(args.loop_interval)
    else:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "infer":
                elapsed, text = run_inference(model, tokenizer, device)
                log(f"INFER_DONE elapsed_s={elapsed:.3f} text={text!r}")
            elif cmd == "exit":
                break


if __name__ == "__main__":
    main()
