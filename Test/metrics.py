import matplotlib.pyplot as plt

train_loss = train_losses
test_loss = val_losses


plt.plot(range(epochs), train_loss, label  ="train loss", color = "green", marker="x", markersize=3)
plt.plot(range(epochs), test_loss, label  = "test loss",  color = "blue", marker = "x", markersize=3)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("train loss vs test loss")
plt.legend()
plt.show()



test_acc = val_accuracies

plt.figure(figsize=(8,5))
plt.plot(range(epochs), test_acc, label="test accuracy", color="red", marker="x", markersize=3)
plt.xlabel("epcohs")
plt.ylabel("accuracy")
plt.title("test accuracy")
plt.legend()
plt.show()
