import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
import src.dataloader as project_dataloader
import Models.Model as models

def setup_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_drml_benchmark(input_file_path, output_dir_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "CAllGraph", "ZeroFeature160-6L_top60_alphe_0.2_newDiseaseFeature_", "Model", "bestmodel_0.plt")
    dataset_root = os.path.join(script_dir, "CAllGraph")
    
    # Model hyperparameters
    params = {
        "feature_dim": 160,
        "top_k": 60,
        "alpha": 0.2,
        "layer_num": 6,
        "seed": 0
    }
    
    setup_seed(params["seed"])
    device = "cuda:0"
    target_disease = "D104300"

    # Load drug IDs
    df = pd.read_csv(input_file_path, header=None)
    drug_ids = [str(row[0]).strip() for _, row in df.iterrows() if pd.notna(row[0])]

    if not drug_ids:
        print("Error: No drug IDs loaded.")
        sys.exit(1)

    # Load graph
    hgm_args = argparse.Namespace(datasetRoot=dataset_root, featureDim=params["feature_dim"], topK=params["top_k"], alphe=params["alpha"], LayerNumber=params["layer_num"], seed=params["seed"])
    hgm = project_dataloader.HeterogeneousGraphManager(hgm_args, device)
    
    graph_info = [
        project_dataloader.GraphInfo(f"{dataset_root}/DTI.csv", "Drug", "interaction_DT", "Protein"),
        project_dataloader.GraphInfo(f"{dataset_root}/DGI.csv", "Disease", "interaction_DG", "Protein"),
        project_dataloader.GraphInfo(f"{dataset_root}/DrugDisI.csv", "Drug", "treath", "Disease"),
        project_dataloader.GraphInfo(f"{dataset_root}/DTI.csv", "Protein", "interaction_TD", "Drug"),
        project_dataloader.GraphInfo(f"{dataset_root}/DGI.csv", "Protein", "interaction_GD", "Disease")
    ]
    hgm.loadHeterogeneousGraph(graph_info)
    drug_dim = hgm.loadFeature(f"{dataset_root}/DrugFeature.csv", "h0", "Drug", isNorm=False, topk=hgm_args.topK)
    disease_dim = hgm.loadFeature(f"{dataset_root}/DiseaseFeature.csv", "h0", "Disease", isNorm=False, topk=hgm_args.topK)
    
    protein_count = hgm.g.num_nodes('Protein')
    protein_dim = params["feature_dim"]
    hgm.g.nodes['Protein'].data['h0'] = torch.zeros(protein_count, protein_dim).to(device)
    
    # Load model
    loaded_content = torch.load(model_path, map_location=device)

    if isinstance(loaded_content, dict):
        model = models.DRHGTModel(hgm.g, protein_dim, drug_dim, disease_dim, hgm_args.featureDim, LayerNumber=hgm_args.LayerNumber)
        if any(key.startswith('module.') for key in loaded_content.keys()):
            loaded_content = {k.replace('module.', ''): v for k, v in loaded_content.items()}
        model.load_state_dict(loaded_content)
    else:
        model = loaded_content

    model = model.to(device)
    model.eval()

    drug_map = hgm.nodesDic['Drug'][0]
    disease_map = hgm.nodesDic['Disease'][0]

    target_disease_id = disease_map[target_disease]

    drug_internal_ids = [drug_map[drug_id] for drug_id in drug_ids if drug_id in drug_map]

    # Call model
    predictions = []
    with torch.no_grad():
        num_drugs = len(drug_internal_ids)
        drug_tensor = torch.tensor(drug_internal_ids, dtype=torch.long)
        disease_tensor = torch.tensor([target_disease_id] * num_drugs, dtype=torch.long)
        
        sub_graph = hgm.createGraph(drug_tensor, disease_tensor, torch.zeros(num_drugs), ("Drug", "treath", "Disease"))
        scores = model(sub_graph.to(device), hgm_args.alphe)

        id_to_drug_map = {v: k for k, v in drug_map.items()}
        for i, internal_id in enumerate(drug_internal_ids):
            predictions.append({
                "drug_id": id_to_drug_map[internal_id],
                "predicted_score": scores[i].item()
            })

    # Save results
    os.makedirs(output_dir_path, exist_ok=True)
    results_df = pd.DataFrame(predictions)
    results_filename = os.path.join(output_dir_path, "predicted_scores.csv")
    results_df.to_csv(results_filename, index=False)