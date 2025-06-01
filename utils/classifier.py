import torch
import pytorch_lightning as pl
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


BATCH_SIZE = 4
NUM_WORKERS = 2
IMG_SIZE = (480, 480)
NUM_CLASSES = 3
MAX_EPOCHS = 75
LEARNING_RATE = 0.001
CLASS_NAMES = ['normal', 'osteopenia', 'osteoporosis'] 

class OsteoporosisClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 3, learning_rate: float = 0.001):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Lightweight, pretrained model
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.backbone.classifier[1].in_features, out_features=num_classes) # change out_features to match
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # tracking
        self.val_predictions = []
        self.val_targets = []
    
    def forward(self, x):
        return self.backbone(x)

    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()

        # Log per-class predictions to debug
        if batch_idx % 50 == 0:  # Log every 50 batches
            unique_preds, counts = torch.unique(predicted, return_counts=True)
            pred_dist = {int(p): int(c) for p, c in zip(unique_preds, counts)}
            print(f"Batch {batch_idx} predictions: {pred_dist}")
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()

        # Store all predictions and targets
        self.val_predictions.extend(predicted.cpu().numpy())
        self.val_targets.extend(labels.cpu().numpy())
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_predictions) > 0:
            # Calculate per-class accuracy
            cm = confusion_matrix(self.val_targets, self.val_predictions, labels=[0, 1, 2])
            
            # Print validation confusion matrix
            print(f"\nValidation Confusion Matrix:")
            print(cm)
            
            # Calculate per-class recall (which is what we care about)
            class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
            for i in range(3):
                if cm.sum(axis=1)[i] > 0:
                    recall = cm[i, i] / cm.sum(axis=1)[i]
                    print(f"{class_names[i]} recall: {recall:.3f}")
            
            # Calculate balanced accuracy
            per_class_recall = []
            for i in range(3):
                if cm.sum(axis=1)[i] > 0:
                    recall = cm[i, i] / cm.sum(axis=1)[i]
                    per_class_recall.append(recall)
            
            if per_class_recall:
                balanced_acc = np.mean(per_class_recall)
                self.log('val_balanced_acc', balanced_acc, prog_bar=True)
        
        # Clear for next epoch
        self.val_predictions = []
        self.val_targets = []
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        
        return {'predictions': predicted, 'targets': labels}
    
    def predict_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        _, predicted = torch.max(outputs, 1)
        return {'predictions': predicted, 'targets': labels}
    
    def configure_optimizers(self):
        # Simple Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Simple step scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    
def evaluate_model(trainer, model, datamodule):
    """Evaluate and show results"""
    test_results = trainer.test(model, datamodule=datamodule)
    predictions = trainer.predict(model, datamodule=datamodule)
    
    all_predictions = []
    all_targets = []
    
    for batch_pred in predictions:
        all_predictions.extend(batch_pred['predictions'].cpu().numpy())
        all_targets.extend(batch_pred['targets'].cpu().numpy())
    
    class_names = ['Normal', 'Osteopenia', 'Osteoporosis']
    print("\nFinal Test Results:")
    print("=" * 50)
    print(classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0))
    
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Overall Accuracy: {accuracy:.1%}")
    
    # Show confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2])
    print(f"\nConfusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Per-class analysis
    print(f"\nPer-class Performance:")
    for i, class_name in enumerate(class_names):
        if cm.sum(axis=1)[i] > 0:
            recall = cm[i, i] / cm.sum(axis=1)[i]
            precision = cm[i, i] / cm.sum(axis=0)[i] if cm.sum(axis=0)[i] > 0 else 0
            print(f"{class_name:12}: Recall={recall:.1%}, Precision={precision:.1%}")