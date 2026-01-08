import argparse
import math
import torch
import time
from torchinfo import summary
from dataloader import DataLoaderLite
from model import GPTConfig, GPT

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


class CosineDecayLR:

    def __init__(self,
                 optimizer,
                 max_steps: int,
                 warmup_steps: int,
                 max_lr: float = 6e-4):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = max_lr * 0.1

    def step(self, step_num: int):
        if step_num < self.warmup_steps:
            lr = self.max_lr * step_num / self.warmup_steps
        elif step_num > self.max_steps:
            lr = self.min_lr
        else:
            progress = (step_num - self.warmup_steps) / (self.max_steps -
                                                         self.warmup_steps)
            coeff = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * coeff
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr


def main(args):
    train_loader = DataLoaderLite(B=args.batch_size, T=1024)

    torch.manual_seed(11)
    if device == "cuda":
        torch.cuda.manual_seed(11)

    torch.set_float32_matmul_precision(args.matmul_percision)

    model = GPT(
        GPTConfig(vocab_size=args.vocab_size,
                  flash_attention=args.flash_attention))
    model.to(device)
    summary(model, input_size=(train_loader.B, train_loader.T), dtypes=[torch.long])
    # 模型编译
    if args.model_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=3e-4,
                                  betas=(0.9, 0.95),
                                  eps=1e-8)
    lr_scheduler = CosineDecayLR(optimizer,
                                 max_steps=args.max_steps,
                                 warmup_steps=args.warmup_steps,
                                 max_lr=args.max_lr)

    autocast_dtype = torch.float32
    if args.autocast_dtype == "float16":
        autocast_dtype = torch.float16
        scaler = torch.GradScaler()
    elif args.autocast_dtype == "bfloat16":
        autocast_dtype = torch.bfloat16
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            logits, loss = model(x, y)
        if step == 0:
            print(f"logits dtype: {logits.dtype}")

        lr_scheduler.step(step)
        if args.autocast_dtype == "float16":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr_scheduler.lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=50, help="最大训练步数")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=10,
                        help="warmup步数")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="最大学习率")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
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
    parser.add_argument("--vocab_size", type=int, default=50257, help="词汇表大小")
    args = parser.parse_args()

    main(args)
