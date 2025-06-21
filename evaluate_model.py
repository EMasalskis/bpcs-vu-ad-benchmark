import argparse
import pandas as pd
import os
import sys
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def main():
    # Args
    parser = argparse.ArgumentParser(
        description="Calculate AUROC, AUPRC, and P@k for a specified model.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model_type",
        choices=['HNNDTA', 'DRML-Ensemble', 'HNNDTA-S1R', 'HNNDTA-DRD2', 'HNNDTA-BIP'],
        help="Select which model or model-target to evaluate."
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    
    if args.model_type.startswith('HNNDTA'):
        model_output_path = os.path.join(project_root, 'data_output', 'HNNDTA', 'predicted_dta_all_targets.csv')
        id_col = 'Drug_ID'
        
        # Specific score based on choice
        if args.model_type == 'HNNDTA-S1R':
            score_col = 'Predicted_DTA_S1R'
        elif args.model_type == 'HNNDTA-DRD2':
            score_col = 'Predicted_DTA_DRD2'
        elif args.model_type == 'HNNDTA-BIP':
            score_col = 'Predicted_DTA_BIP'
        else:
            score_col = 'MAX_DTA'

    elif args.model_type == 'DRML-Ensemble':
        model_output_path = os.path.join(project_root, 'data_output', 'DRML-Ensemble', 'predicted_scores.csv')
        id_col = 'drug_id'
        score_col = 'predicted_score'
        
    positive_set_path = os.path.join(project_root, 'data_input', 'positive_set.csv')

    # Load data
    try:
        positive_df = pd.read_csv(positive_set_path, header=None)
        positive_set = set(positive_df[0].astype(str).str.strip())
    except FileNotFoundError:
        print(f"Error: Positive set file not found at '{positive_set_path}'.")
        sys.exit(1)

    try:
        predictions_df = pd.read_csv(model_output_path)
    except FileNotFoundError:
        print(f"Error: Model output file not found at: '{model_output_path}'.")
        sys.exit(1)
        
    # Prep data for scikit-learn
    y_true = []
    y_scores = []
    all_drugs = []
    for _, row in predictions_df.iterrows():
        drug_id = str(row[id_col]).strip()
        all_drugs.append((drug_id, row[score_col]))
        y_true.append(1 if drug_id in positive_set else 0)
        y_scores.append(row[score_col])

    # AUROC and AUPRC calculations
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    # Precision@k calculation
    sorted_drugs = sorted(all_drugs, key=lambda x: x[1], reverse=True)
    k = 20
    top_k_drugs = [drug_id for drug_id, score in sorted_drugs[:k]]
    positives_in_top_k = sum(1 for drug_id in top_k_drugs if drug_id in positive_set)
    p_at_k = positives_in_top_k / k if k > 0 else 0

    # Print results
    print(f"Evaluation for {args.model_type}:")
    print(f"Total drugs: {len(y_true)}")
    print(f"Positive samples found: {sum(y_true)} / {len(positive_set)}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print(f"Precision@{k}: {p_at_k:.4f} ({positives_in_top_k}/{k})")


if __name__ == "__main__":
    main()