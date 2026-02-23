import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np

def load_processed_data(folder='../data/c'):
    # load 
    X_train = np.load(f'{folder}/X_train_csp.npy')
    y_train = np.load(f'{folder}/y_train.npy')
    X_test = np.load(f'{folder}/X_test_csp.npy')
    y_test = np.load(f'{folder}/y_test.npy')
    
    print(f"Successfully loaded processed data from {folder}")

    return X_train, y_train, X_test, y_test


def print_classification_metrics(y_true, y_pred, model_name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    
    return acc, prec, rec, f1

def plot_multiple_confusion_matrices(models_dict, X_test, y_test):
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    # Handle case where only 1 model is passed
    if n_models == 1:
        axes = [axes]
        
    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left Hand', 'Foot'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name}\nConfusion Matrix')
        
    plt.tight_layout()
    plt.show()

def plot_combined_roc_curve(models_dict, X_test, y_test):

    plt.figure(figsize=(8, 6))
    
    for name, model in models_dict.items():
        # Class 1
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models that use decision_function instead (like some SVMs)
            y_score = model.decision_function(X_test)
            
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()