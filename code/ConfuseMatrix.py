import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
from DataTransformer import data_factory

from keras.utils.np_utils import *
from sklearn.model_selection import train_test_split

train_data_path = r'D:/JupyterProject/digit-recognizer/train.csv'
test_data_path = r'D:/JupyterProject/digit-recognizer/test.csv'

# Look at confusion matrix
# Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels
    errors_index：错误预测值索引集合
    img_errors：错误值的图像集合
    pred_errors：
    obs_errors：
    """
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28,28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()

digit_data = data_factory(train_data_path, test_data_path)
train_data = digit_data.train_data
X = digit_data.train_data.drop(columns='label').values / 255
X = torch.from_numpy(X).view(-1, 1, 28, 28)
y = digit_data.train_data.label.values
y = torch.from_numpy(y).type(torch.LongTensor)

X_train, X_test, y_val, y_true = train_test_split(X,y,test_size=0.2, random_state=42, shuffle=False)

# y_true_class = to_categorical(y_true, 10)  # 相当于y_train  测试集真实标签值one_hot编码


model = torch.load("Net.pth")
with torch.no_grad():
    y = model(X_test)

y_pred_class = np.argmax(y, axis=1)  # 预测分类值，argmax返回最大值索引


errors = (y_pred_class - y_true != 0)  # errors_index
Y_pred_classes_errors = y_pred_class[errors]  # 错误预测的预测结果  集合
Y_pred_errors = y[errors]  # 返回预测概率
Y_true_errors = y_true[errors]  # 错误预测真实值
X_val_errors = X_test[errors]  # 错误预测图片

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors.detach().numpy(), axis=1)  # 错误预测概率最大值

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))
"""
Y_pred_errors中真实数字的索引对应的概率位置放到对角线位置后，提取对角线元素。
diagonal参数axis1，axis2确定一个平面，在该平面上取对角元素
"""
# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors    # 真实值和预测值概率的插值

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors
most_important_errors = sorted_dela_errors[-6:]  # 索引排序

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
