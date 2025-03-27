import torch
import numpy as np

from scripts.get_pwms import load_pwms, preprocess
from scripts.helper import set_seed
from scripts.data_loader import get_k_folds
from scripts.train_eval import train_CV
from scripts.helper import load_models
from scripts.plots import total_accuracy, total_class_accuracy, total_conf_matrix, total_confidence, IG, confidence, class_accuracy_comp

def main():
    
    ### Initialize seed, device, model_name ###
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = '03_26_20fold'

    ### Load data ###
    motifs = load_pwms()
    pwms, fams, m_ids, id_to_fam, id_to_fam_cnts, n_fams = preprocess(motifs, print_info=False)
    k_folds = get_k_folds(pwms,fams,k=20)

    bhlh = [m_id for i,m_id in enumerate(m_ids) if fams[i] == 3]

    ### Train Model ###
    
    # Make new directory: models/saves/{model_name}
    # models = train_CV(pwms,fams,k_folds,device,model_name)

    ### Load Model ###
    models = load_models(model_name,device,k=20)

    ### Evaluation ###
    # accuracies = total_accuracy(pwms,fams,models,k_folds,model_name)
    # total_class_accuracy(pwms,fams,models,k_folds,model_name,id_to_fam_cnts)
    # total_conf_matrix(pwms,fams,models,k_folds,model_name,id_to_fam_cnts)
    # total_confidence(pwms,fams,models,k_folds,model_name,True)

    for m in bhlh:
        IG(m, pwms, fams, m_ids, models, k_folds, model_name, id_to_fam, device,True)


if __name__ == '__main__':
    main()