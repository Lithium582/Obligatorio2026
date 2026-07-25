import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    model,
    X,
    y,
    threshold=0.5,
    title="Matriz de confusión",
):
    """Grafica una matriz de confusión aplicando un umbral explícito."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot()
    plt.title(title)
    plt.show()
    return display
