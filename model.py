import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class DocumentClassifier(nn.Module):
    def __init__(self):
        super(DocumentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((128, 128))
        self.fc1 = nn.Linear(32 * 128 * 128, 100)
        self.fc2 = nn.Linear(100, 2)  # 2 output classes (documents and non-documents)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the images to 512x512
    transforms.ToTensor(),          # Convert images to tensors
])

train_data = datasets.ImageFolder('data', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)

model = DocumentClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.4f}')
        running_loss = 0.0

print('Finished Training')

# Save the model for future use
torch.save(model.state_dict(), 'document_classifier_model.pth')