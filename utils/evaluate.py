import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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