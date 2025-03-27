import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def accuracy(save=False):

    pcc = [0.8296261239943209, 0.8263132986275438, 0.8149550402271651, 0.8102224325603408, 0.8017037387600567, 0.795551348793185]
    euclid = [0.8400378608613346, 0.8386180785612872, 0.8310459062943681, 0.8253667770941789, 0.8154283009938476, 0.8050165641268339]
    rmse = [0.836251774727875, 0.8386180785612872, 0.8338854708944629, 0.8258400378608614, 0.8159015617605301, 0.807382867960246]
    ensmbl = [0.8419309039280644, 0.8400378608613346, 0.8334122101277804, 0.8267865593942262, 0.8159015617605301, 0.8116422148603881]

    x = range(1,12,2)
    
    plt.plot(x, [y*100 for y in pcc], color='blue',label='PCC')
    plt.plot(x, [y*100 for y in euclid], color='orange',label='Euclidean Dist.')
    plt.plot(x, [y*100 for y in rmse], color='green',label='2-RMSE')
    plt.plot(x, [y*100 for y in ensmbl], color='red',label='KNN Ensemble')

    plt.xlabel('K')
    plt.ylabel('Accuracy (%)')
    plt.ylim(79,85)
    plt.xticks(range(1,12,2))
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/accuracy.png')
        plt.close()

def accuracy_FCN(save=False):
    
    categories = ['Deep Learning', 'PCC', 'Euclidean Dist.', '2-RMSE', 'KNN Ensemble']
    values = [82.25651392632527, 82.96261239943209, 84.00378608613346, 83.6251774727875, 84.19309039280644]
    colors = ['purple', 'gray', 'gray', 'gray', 'gray']
    errors = [3.7121765900132546,0,0,0,0]

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, yerr=errors, color=colors, capsize=2)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,100)
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/accuracy_FCN.png')
        plt.close()

def time_FCN(save=False):
    
    categories = ['Deep Learning', 'PCC', 'Euclidean Dist.', '2-RMSE', 'KNN Ensemble']
    values = [2.3, 3576, 2096, 1990, 7653]
    colors = ['purple', 'gray', 'gray', 'gray', 'gray']

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color=colors)
    plt.yscale('log')
    plt.xlabel("Model")
    plt.ylabel("Time (seconds, log scale)")
    
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/time_FCN.png')
        plt.close()

def conf_matrix(y_pred,fams,save=False):

    cm = confusion_matrix(fams, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    epsilon = 1e-10
    cm = cm / (row_sums + epsilon)


    plt.figure(figsize=(12,10))
    sns.heatmap(cm*100, annot=True, fmt='.1f', cmap="Blues", cbar=False)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    yticks = np.array(range(20)) + 0.5
    plt.yticks(yticks, range(20), rotation=0)
    plt.tight_layout()
    if not save:
        plt.show()
    else:
        plt.savefig(f'outputs/conf_matrix.png')
        plt.close()