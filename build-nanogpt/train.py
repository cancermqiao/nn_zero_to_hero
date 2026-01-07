import argparse
import torch
import time
from dataloader import DataLoaderLite
from model import GPTConfig, GPT

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


def main(args):
    train_loader = DataLoaderLite(B=16, T=1024)

    torch.manual_seed(11)
    if device == "cuda":
        torch.cuda.manual_seed(11)

    torch.set_float32_matmul_precision(args.matmul_percision)

    model = GPT(GPTConfig(vocab_size=args.vocab_size, flash_attention=args.flash_attention))
    model.to(device)
    # 模型编译
    if args.model_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    autocast_dtype = torch.float32
    if args.autocast_dtype == "float16":
        autocast_dtype = torch.float16
        scaler = torch.GradScaler()
    elif args.autocast_dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            logits, loss = model(x, y)
        if i == 0:
            print(f"logits dtype: {logits.dtype}")
        
        if args.autocast_dtype == "float16":
            scaler.scale(loss)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul_percision",
                        type=str,
                        default="highest",
                        choices=["highest", "high", "medium"],
                        help="矩阵计算精度")
    parser.add_argument("--autocast_dtype",
                        type=str,
                        default="float32",
                        choices=["float16", "bfloat16", "float32"],
                        help="autocast数据类型")
    parser.add_argument("--model_compile",
                        action="store_true",
                        default="False",
                        help="模型编译")
    parser.add_argument("--flash_attention",
                        action="store_true",
                        default="False",
                        help="是否使用flash attention")
    parser.add_argument("--vocab_size",
                        type=int,
                        default=50257,
                        help="词汇表大小")
    args = parser.parse_args()

    main(args)
