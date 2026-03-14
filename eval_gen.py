import time, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_quantization

def benchmark(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {args.model} to {device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
    model = model.cuda()
    model, _ = model_quantization(model, args.model, 4, 4)

    prompt = "Summarize the following text in one sentence:\n" + ("The quick brown fox jumps over the lazy dog. " * 100)
    inputs = tokenizer(
        [prompt] * args.batch_size,
        max_length=args.input_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                use_cache=True
            )
            torch.cuda.synchronize()

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True
        )
        torch.cuda.synchronize()
        first_token_latency = (time.time() - t0) * 1000.0

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.time()
        _ = model.generate(
            **inputs,
            max_new_tokens=args.new_tokens,
            do_sample=False,
            use_cache=True
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

    total_tokens = args.batch_size * args.new_tokens
    tps = total_tokens / elapsed

    print(f"\n=== Generation ===")
    print(f"batch_size={args.batch_size}, input_len={args.input_len}, new_tokens={args.new_tokens}")
    print(f"First-token latency: {first_token_latency:.1f} ms (per batch)")
    print(f"Steady-state throughput: {tps:.1f} tokens/s")
    print(f"Elapsed: {elapsed:.2f}s for {total_tokens} tokens")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-2.7b")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_len", type=int, default=256)
    parser.add_argument("--new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    benchmark(args)
