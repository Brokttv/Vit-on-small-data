from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()
y_true = []
y_pred = []

with torch.inference_mode():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        predicted_classes = preds.argmax(dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted_classes.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()
