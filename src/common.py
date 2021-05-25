import numpy as np
from PIL import ImageFile
from matplotlib import pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    #
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.show()

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    print(f"TP: {TP}, TN: {TN} FP: {FP}, FN: {FN}")
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    print(f"Precision score: {precision:.2f}, Recall score: {recall:.2f}, F1 score: {f1_score:.2f}")


def calculate_accuracy_metrics(cm):
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    print(f"TP: {TP}, TN: {TN} FP: {FP}, FN: {FN}")

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    print(f"Precision score: {precision:.2f}, Recall score: {recall:.2f}, F1 score: {f1_score:.2f}")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }
