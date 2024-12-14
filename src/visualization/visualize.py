import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def fig_confusion_matrix(y_test, y_pred, file_path):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues")
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités')
    plt.title('Matrice de confusion')
    plt.savefig(file_path)
    plt.close()
    print(f"Confusion matrix saved to {file_path}")
