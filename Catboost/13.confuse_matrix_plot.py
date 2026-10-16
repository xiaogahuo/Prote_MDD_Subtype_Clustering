
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from SUtils import *
import numpy as np

def get_score_label(exp, model_name, d_type):
    # get the raw scores and labels from the test csvfile for the ROC PR curves
    df = pd.read_csv(result_path + "{}/{}_{}_eval.csv".format(exp, model_name, d_type))
    return df[['pred_label']].values, df['label'].values

def draw_confuse(exp, model_name):
    lw = 2
    types = ['train', 'test', 'infer']
    # fig, axs = plt.subplots(3, 1, figsize=(8, 15), gridspec_kw={'height_ratios': [1, 1, 1]})
    fig, ax = plt.subplots(2, 1, figsize=(8, 15), gridspec_kw={'height_ratios': [1, 1]})
    for k in range(2):
      pred_label, labels = get_score_label(exp, model_name, types[k])
      # pred_label=np.argmax(np.array(scores), axis=1)
      matrix = confusion_matrix(np.array(labels), np.array(pred_label))
      sns.heatmap(matrix, annot=True, fmt='d', cmap='Oranges',
                  xticklabels=['HC', "MDD"],
                  yticklabels=['HC', "MDD"], ax=ax[k])
      ax[k].set_xlabel('Predicted')
      ax[k].set_ylabel('Actual')
      ax[k].set_title('{}-Confusion Matrix'.format(types[k]))
    if not os.path.exists(img_path + "{}/".format(exp)):
        os.makedirs(img_path + "{}/".format(exp))
    plt.savefig(img_path + "{}/{}_{}_confusion_martix.png".format(exp, exp, model_name), bbox_inches='tight')
    plt.clf()
    plt.close()


if __name__ == "__main__":
    model_name = "catboost"
    draw_confuse("0_VS_12", model_name)
    draw_confuse("0_VS_1", model_name)
    draw_confuse("0_VS_2", model_name)

    # draw_confuse("cluster", '1&2', 'Prote_cols_subtype1')
    # draw_confuse("cluster", '1', 'Prote_cols_subtype1')