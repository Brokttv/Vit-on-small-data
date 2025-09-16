import numpy as np
seed= 42

torch.manual_seed(seed) # for cpu operations
torch.cuda.manual_seed(seed) # for GPU opearations
np.random.seed(seed) # for numpy operations



def train(dataloader, model, loss_fn, optim, scheduler, device):
    model.train()
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss=0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2, device=device)

        optim.zero_grad()
        preds = model(x)
        loss =  mixup_criterion(loss_fn, preds, y_a, y_b, lam)
        train_loss+=loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        if batch_idx % 100 == 0:
            num_samples_processed = (batch_idx + 1) * len(x)
            print(f"loss: {loss.item():.4f} | samples_processed: {num_samples_processed} / {total_samples}")

    train_loss/=num_batches
    return train_loss
