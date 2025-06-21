import numpy as np
import pandas as pd
import os
import sys

from my_utils import predict_DTA

import warnings
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


def run_hnndta_benchmark(input_path, output_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    id_col = 0
    smiles_col = 1

    # Load drug IDs and SMILES
    drug_ids = []
    drug_smiles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) > max(id_col, smiles_col):
                drug_ids.append(parts[id_col])
                drug_smiles.append(parts[smiles_col])
            else:
                print(f"Warning: Skipping line {ln+1}.")

    if not drug_smiles:
        print("Error: No SMILES loaded.")
        sys.exit(1)

    # Load models
    models_file = os.path.join(script_dir, "tmp_files", "top_models_pretrained.csv")
    try:
        model_configs_df = pd.read_csv(models_file, header=None)
        drug_encodings = model_configs_df[0].values
        target_encodings = model_configs_df[1].values
    except FileNotFoundError:
        print(f"Error: Could not load models from {models_file}.")
        sys.exit(1)

    path_S1R = os.path.join(script_dir, "saved_models", "saved_models_S1R_t")
    path_DRD2 = os.path.join(script_dir, "saved_models", "saved_models_DRD2_t")
    path_BIP = os.path.join(script_dir, "saved_models", "saved_models_BIP_t")

    # Call predict_DTA
    result_scores = predict_DTA(drug_encodings, target_encodings,
                              path_S1R, path_DRD2, path_BIP,
                              drug_smiles, saved_output=False)
    
    # Save all results
    average_scores = np.average(result_scores, axis=0)
    target_names = ["S1R", "DRD2", "BIP"]
    predictions_df = pd.DataFrame({'Drug_ID': drug_ids, 'SMILES': drug_smiles})

    for i, target_name in enumerate(target_names):
        predictions_df[f'Predicted_DTA_{target_name}'] = average_scores[i, :]
        
    dta_columns = [f'Predicted_DTA_{name}' for name in target_names]
    predictions_df['MAX_DTA'] = predictions_df[dta_columns].max(axis=1)
    
    output_columns = ['Drug_ID'] + dta_columns + ['MAX_DTA']
    
    output_df = predictions_df[output_columns]
    output_file_all = os.path.join(output_dir, "predicted_dta_all_targets.csv")
    output_df.to_csv(output_file_all, index=False)

    # Save top drugs for each target
    top_n = 20
    for i, target_name in enumerate(target_names):
        average_scores_target = average_scores[i, :]
        sorted_drug_ids_for_target = np.argsort(average_scores_target)[::-1]
        
        top_ids = [drug_ids[j] for j in sorted_drug_ids_for_target[:top_n]]
        top_smiles = [drug_smiles[j] for j in sorted_drug_ids_for_target[:top_n]]
        top_scores = average_scores_target[sorted_drug_ids_for_target[:top_n]]
        
        df_top_target = pd.DataFrame({
            'Drug_ID': top_ids,
            'SMILES': top_smiles,
            'Predicted_DTA': top_scores
        })
        output_file_target = os.path.join(output_dir, f"top_{top_n}_for_{target_name}.csv")
        df_top_target.to_csv(output_file_target, index=False)
    