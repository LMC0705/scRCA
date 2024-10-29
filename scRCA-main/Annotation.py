import torch
import torch.nn as nn
import torch.optim as optim
import anndata as ad
from torch.utils.data import DataLoader
from processing_celldata import CellDataset
from model import cellNet  # Assuming a cellNet model file is provided
from loss import loss_coteaching

class scRCAAnnotator:
    def __init__(self, model_f, model_g, learning_rate=0.01, n_epoch=20, noise_rate=0.45, forget_rate=0.35):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.noise_rate = noise_rate
        self.forget_rate = forget_rate
        self.model_f = model_f
        self.model_g = model_g
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_f = optim.Adam(self.model_f.parameters(), lr=self.learning_rate)
        self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.learning_rate)
    
    def pretrain(self, train_loader, epochs=30):
        # Pretrain both models using the reference data
        for epoch in range(epochs):
            self.model_f.train()
            self.model_g.train()
            for i, (input, target, _) in enumerate(train_loader):
                input, target = input.cuda(), target.cuda()
                
                # Train model_f
                self.optimizer_f.zero_grad()
                output_f = self.model_f(input)
                loss_f = self.criterion(output_f, target)
                loss_f.backward()
                self.optimizer_f.step()
                
                # Train model_g
                self.optimizer_g.zero_grad()
                output_g = self.model_g(input)
                loss_g = self.criterion(output_g, target)
                loss_g.backward()
                self.optimizer_g.step()

    def train_and_annotate(self, train_loader, query_loader):
        # Main training function with co-teaching strategy
        for epoch in range(self.n_epoch):
            self.model_f.train()
            self.model_g.train()
            for i, (cells, labels, _) in enumerate(train_loader):
                cells, labels = cells.cuda(), labels.cuda()
                
                # Forward pass for both models
                logits_f = self.model_f(cells)
                logits_g = self.model_g(cells)
                
                # Co-teaching loss calculation
                loss_f, loss_g, _, _ = loss_coteaching(logits_f, logits_g, labels, self.forget_rate)
                
                # Update model_f
                self.optimizer_f.zero_grad()
                loss_f.backward()
                self.optimizer_f.step()
                
                # Update model_g
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_g.step()

        # Annotate query data with model_f after training
        predictions = []
        self.model_f.eval()
        with torch.no_grad():
            for input, _ in query_loader:
                input = input.cuda()
                output_f = self.model_f(input)
                _, pred = torch.max(output_f, 1)
                predictions.extend(pred.cpu().numpy())
        return predictions

def scRCA_annotate(refer_data_path, query_data_path, learning_rate=0.01, n_epoch=20, noise_rate=0.45, forget_rate=0.35, pretrain_epochs=30):
    # Load reference and query data
    refer_data = ad.read_h5ad(refer_data_path)
    query_data = ad.read_h5ad(query_data_path)
    
    # Extract cell type labels from reference data and encode them
    cell_types = refer_data.obs['cell_types'].astype('category').cat.codes

    # Create datasets and data loaders for reference and query data
    refer_dataset = CellDataset(refer_data.X, cell_types)
    refer_loader = DataLoader(refer_dataset, batch_size=256, shuffle=True)

    query_dataset = CellDataset(query_data.X)
    query_loader = DataLoader(query_dataset, batch_size=256, shuffle=False)
    
    # Initialize two models for co-teaching
    model_f = cellNet(input_dim=refer_data.X.shape[1], num_classes=len(set(cell_types))).cuda()
    model_g = cellNet(input_dim=refer_data.X.shape[1], num_classes=len(set(cell_types))).cuda()
    
    # Initialize annotator with specified parameters
    annotator = scRCAAnnotator(model_f, model_g, learning_rate=learning_rate, n_epoch=n_epoch, noise_rate=noise_rate, forget_rate=forget_rate)
    
    # Pretrain models with specified epochs
    if pretrain_epochs > 0:
        annotator.pretrain(refer_loader, epochs=pretrain_epochs)
    
    # Train models and annotate query data
    predictions = annotator.train_and_annotate(refer_loader, query_loader)
    
    # Add predicted cell types to query data's observation data
    query_data.obs['predicted_cell_types'] = predictions
    
    return query_data