"""Test PyTorch Lightning batch size warning."""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return a list of tensors (similar to our collate function)
        image = torch.randn(3, 224, 224)
        target = {'labels': torch.tensor([1, 2, 3])}
        return image, target


def collate_fn(batch):
    """Similar to our custom collate function."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets


class SimpleLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 1)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        print(f"Batch {batch_idx}: {len(images)} images")
        loss = torch.tensor(0.0, requires_grad=True)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        print(f"Val batch {batch_idx}: {len(images)} images")
        return torch.tensor(0.0)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():
    # Create data
    train_dataset = SimpleDataset(100)
    val_dataset = SimpleDataset(50)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create module and trainer
    model = SimpleLightningModule()
    
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='cpu',
        num_sanity_val_steps=2,
        log_every_n_steps=1
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()