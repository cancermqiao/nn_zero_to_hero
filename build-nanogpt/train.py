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
    train_loader = DataLoaderLite(B=args.batch_size, T=args.sequence_length)

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

    optimizer = model.configure_optimizers(weight_decay=0.1,
                                           learning_rate=args.max_lr,
                                           device=device)
    lr_scheduler = CosineDecayLR(optimizer,
                                 max_steps=args.max_steps,
                                 warmup_steps=args.warmup_steps,
                                 max_lr=args.max_lr)

    autocast_dtype = torch.float32
    if args.autocast_dtype == "bfloat16":
        autocast_dtype = torch.bfloat16

    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        optimizer.zero_grad()
        lr_scheduler.step(step)
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                logits, loss = model(x, y)
            loss = loss / args.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        if step == 0:
            print(f"logits dtype: {logits.dtype}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(
            f"step {step:4d} | loss: {loss_accum:.6f} | lr: {lr_scheduler.lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=50, help="最大训练步数")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=10,
                        help="warmup步数")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="最大学习率")
    parser.add_argument("--global_batch_size", type=int, default=524288, help="全局batch size")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--sequence_length", type=int, default=1024, help="序列长度")
    parser.add_argument("--matmul_percision",
                        type=str,
                        default="highest",
                        choices=["highest", "high", "medium"],
                        help="矩阵计算精度")
    parser.add_argument("--autocast_dtype",
                        type=str,
                        default="float32",
                        choices=["bfloat16", "float32"],
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

    assert args.global_batch_size % (args.batch_size * args.sequence_length) == 0, "全局batch size必须是batch size和序列长度的整数倍"
    args.grad_accum_steps = args.global_batch_size // (args.batch_size * args.sequence_length)
    main(args)
