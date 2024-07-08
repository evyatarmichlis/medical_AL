import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets

import torch_geometric.nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define CNN for Euclidean features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Transformation and Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset
dataset = datasets.ImageFolder('data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model_save_path = 'dgcl_model.pth'

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.resnet(x)


# GCN Feature Extractor
class GCNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNFeatureExtractor, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# DGCL Model
class DGCL(nn.Module):
    def __init__(self, cnn, gcn, feature_dim):
        super(DGCL, self).__init__()
        self.cnn = cnn
        self.gcn = gcn
        self.fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, x, edge_index):
        cnn_features = self.cnn(x)
        gcn_features = self.gcn(cnn_features, edge_index)
        combined_features = torch.cat([cnn_features, gcn_features], dim=1)
        return self.fc(combined_features)


# Memory Store
class MemoryStore:
    def __init__(self, num_samples, feature_dim, num_clusters):
        self.representations = torch.zeros(num_samples, feature_dim)
        self.centroids = torch.zeros(num_clusters, feature_dim)
        self.labels = torch.zeros(num_samples, dtype=torch.long)

    def update_representations(self, indices, new_representations):
        self.representations[indices] = new_representations

    def update_centroids(self):
        for i in range(self.centroids.size(0)):
            cluster_indices = (self.labels == i).nonzero(as_tuple=True)[0]
            if len(cluster_indices) > 0:
                self.centroids[i] = self.representations[cluster_indices].mean(dim=0)

    def get_pseudo_labels(self, features):
        distances = torch.cdist(features, self.centroids)
        return distances.argmin(dim=1)


# Initialize memory store
num_samples = len(dataset)
feature_dim = 512  # Should match the output feature dimension of the DGCL model
num_clusters = 5
memory_store = MemoryStore(num_samples, feature_dim, num_clusters)


# Create adjacency matrix
def create_adjacency_matrix(images, k=4):
    features = images.view(images.size(0), -1).detach().cpu().numpy()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, np.arange(features.shape(0)))
    adjacency_matrix = knn.kneighbors_graph(features).toarray()
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    return edge_index


# Dynamic clustering step
def dynamic_clustering_step(model, dataloader, memory_store):
    model.eval()
    all_features = []
    all_indices = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            indices = torch.arange(batch_idx * images.size(0), (batch_idx + 1) * images.size(0))
            edge_index = create_adjacency_matrix(images)
            features = model(images, edge_index)
            all_features.append(features)
            all_indices.append(indices)
    all_features = torch.cat(all_features, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    # Update memory store representations
    memory_store.update_representations(all_indices, all_features)

    # Update centroids
    memory_store.update_centroids()

    # Get pseudo-labels
    pseudo_labels = memory_store.get_pseudo_labels(all_features)

    return pseudo_labels


# Consistency smoothing step
def consistency_smoothing_step(memory_store, lambda_factor=0.6):
    memory_store.representations = lambda_factor * memory_store.representations + \
                                   (1 - lambda_factor) * memory_store.representations


# Training function
def train_model(dataloader, model, optimizer, criterion, memory_store, num_epochs, initial_lr, lambda_factor):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lambda epoch: 1 if epoch < 30 else 0.1 ** ((epoch - 30) / 70))
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            indices = torch.arange(batch_idx * images.size(0), (batch_idx + 1) * images.size(0))
            edge_index = create_adjacency_matrix(images)
            outputs = model(images, edge_index)

            # Dynamic clustering and update memory store
            if epoch % 10 == 0 and batch_idx == 0:
                pseudo_labels = dynamic_clustering_step(model, dataloader, memory_store)

            # Consistency smoothing
            consistency_smoothing_step(memory_store, lambda_factor)

            # Get pseudo-labels for the current batch
            batch_pseudo_labels = pseudo_labels[indices]

            loss = criterion(outputs, batch_pseudo_labels.to(device))
            loss.backward()
            optimizer.step()

            memory_store.update_representations(indices, outputs.detach().cpu())

        # Adjust learning rate
        scheduler.step()
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


# Model, optimizer, and criterion initialization
cnn = CNNFeatureExtractor().to(device)
gcn = GCNFeatureExtractor(in_channels=2048, hidden_channels=256).to(device)
model = DGCL(cnn, gcn, feature_dim=256).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner
    model = nn.DataParallel(model)



optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()




# Training the model
train_model(dataloader, model, optimizer, criterion, memory_store, num_epochs=100, initial_lr=0.01, lambda_factor=0.6)



def extract_features(dataloader, model):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            edge_index = create_adjacency_matrix(images)
            features = model(images, edge_index)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of DGCL extracted features')
    plt.show()



model = DGCL(cnn, gcn, feature_dim=256).to(device)
model.load_state_dict(torch.load('dgcl_model.pth'))
print("Model loaded from dgcl_model.pth")

# Extract features using the trained DGCL model
features, labels = extract_features(dataloader, model)

# Visualize the features using t-SNE
visualize_tsne(features, labels)