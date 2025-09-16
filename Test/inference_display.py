fig=plt.figure(figsize=(19,7))
model.eval()
classes = train_data.classes

rows,cols= 5,5

with torch.inference_mode():
  for i in range(1,rows*cols+1):
    random_idx = torch.randint(0,len(test_data), size=[1]).item()
    x,y= test_data[random_idx]

    x= x.unsqueeze(dim=0).to(device)
    preds = model(x)
    predicted_class = preds.argmax(dim=1).item()

    fig.add_subplot(rows,cols,i)
    plt.imshow(x.squeeze().permute(1,2,0).cpu())
    title_text = f"Predicted class:{classes[predicted_class]}  |  True label:{classes[y]}"

    if predicted_class==y:
      plt.title(title_text, fontsize=10,c="g")
    else:
      plt.title(title_text, fontsize=10, c="r")

    plt.axis(False)
plt.tight_layout()
plt.show()
