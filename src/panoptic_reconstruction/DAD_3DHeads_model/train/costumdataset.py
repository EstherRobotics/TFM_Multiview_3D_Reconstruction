
import torch 



# Define una clase para tu conjunto de datos personalizado
class CustomDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(256, 256), label_size=(1, 5023, 2)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.label_size = label_size

        self.images = torch.randn(num_samples, 3, *image_size)
        self.labels = torch.randn(num_samples, *label_size, requires_grad=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label



train_dataset = CustomDataset(num_samples=50, image_size=(256, 256), label_size=(5023, 2))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


