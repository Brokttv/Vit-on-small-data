def test(dataloader, model, loss_fn, device):
    model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct_preds = 0, 0

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x)
            test_loss += loss_fn(preds, y).item()
            probs = torch.softmax(preds, dim=1)
            correct_preds += (probs.argmax(dim=1) == y).sum().item()

    test_loss /= num_batches
    acc = (correct_preds / num_samples) * 100
    print(f"average_test_loss: {test_loss:.4f}  |  accuracy: {acc:.2f}%")

    return test_loss, acc
