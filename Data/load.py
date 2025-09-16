train_data = datasets.CIFAR10(root="dataset",
                               train=True,

                               download=True,
                               transform= transform_train)

test_data = datasets.CIFAR10(root="dataset",
                                train = False,

                                download=True,
                                transform = transform_test
                                )
num_workers = os.cpu_count()
batch_size = 32

train_dataloader =  DataLoader(train_data, num_workers= num_workers, batch_size=batch_size,pin_memory=True, shuffle =True)
test_dataloader = DataLoader (test_data, num_workers=num_workers, batch_size=batch_size, pin_memory=True, shuffle = False)
