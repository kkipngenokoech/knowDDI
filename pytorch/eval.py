import argparse
import torch
from dataset import SubgraphDataset
from evaluator import Evaluator_multiclass, Evaluator_multilabel

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--db_path', type=str, required=True, help='Path to the database')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth file')
# Add all the other arguments as in the active file
# ...
params = parser.parse_args()

# Load the model
model = torch.load(params.model_path)

# Load the evaluation data
eval_data = SubgraphDataset(db_path=params.db_path,
                            db_name='eval_subgraph',
                            use_pre_embeddings=params.use_pre_embeddings,
                            dataset=params.dataset,
                            kge_model=params.kge_model,
                            ssp_graph=train_data.ssp_graph,
                            id2entity=train_data.id2entity,
                            id2relation=train_data.id2relation,
                            rel=train_data.num_rels,
                            global_graph=train_data.global_graph,
                            dig_layer=params.num_dig_layers,
                            BKG_file_name=params.BKG_file_name)

# Initialize the appropriate evaluator
evaluator = Evaluator_multiclass(params, model, eval_data) if params.dataset == 'drugbank' \
    else Evaluator_multilabel(params, model, eval_data)

# Evaluate model
evaluation_result = evaluator.evaluate()
print(evaluation_result)