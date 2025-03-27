

from scripts.functions import set_seed, run_knn, read_tsv, get_target_dict, load_pwms, preprocess, total_knn
from scripts.plots import accuracy, accuracy_FCN, time_FCN, conf_matrix
import matplotlib.pyplot as plt
import pickle

def main():

    set_seed(42)

    
    ### Show Plots ###
       
    
    # Run for MEME_all.tsv first

    motifs = load_pwms()
    pwms, fams, m_ids, id_to_fam, fam_to_id, main_classes = preprocess(motifs, print_info=False)

    ### Obtain df ###

    # tomtom_df1 = read_tsv('data/MEME_all.tsv')
    # tomtom_df2 = read_tsv('data/MEME_all2.tsv')
    # tomtom_df3 = read_tsv('data/MEME_all3.tsv')
    # target_dict = get_target_dict(m_ids)
    
    
    # accuracy, y_pred = run_knn(1, tomtom_df, target_dict, main_classes, m_ids, fams, fam_to_id)
    # pcc, euclid, rmse, ensmbl, y_pred = total_knn(1, tomtom_df1, tomtom_df2, tomtom_df3, target_dict, main_classes, m_ids, fams, fam_to_id)

    with open('data/1NN_ensmbl_ypred.pkl', 'rb') as f:
        y_pred = pickle.load(f)

    conf_matrix(y_pred,fams,True)

if __name__ == '__main__':
    main()