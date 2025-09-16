#Cosine decay scheduler

import math

def lr_scheduler(step, total_steps=78125, warmup_steps=21000):
    if step < warmup_steps:
        return  (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))  

#Loss,optim, and scheduker init

loss_fn= nn.CrossEntropyLoss()
optim = torch.optim.AdamW(
    model.parameters(),
    lr= 1e-3,
    weight_decay=0.001,
    betas=(0.9, 0.999),
    eps=1e-8
)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_scheduler)
