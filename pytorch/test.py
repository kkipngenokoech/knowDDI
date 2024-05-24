import os
import torch
from data_processor.datasets import SubgraphDataset
from train import process_dataset

# Create a simple class to hold parameters
class Params:
    def __init__(self, main_dir, dataset, train_file, valid_file, test_file, BKG_file_name, add_traspose_rels):
        self.main_dir = main_dir
        self.dataset = dataset
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.BKG_file_name = BKG_file_name
        self.add_traspose_rels = add_traspose_rels
        self.use_pre_embeddings = False
        self.kge_model = 'TransE'
        self.num_dig_layers = 3
        self.hop = 2
        self.file_paths = {
            'train': os.path.join(self.main_dir, f'../data/{self.dataset}/{self.train_file}.txt'),
            'valid': os.path.join(self.main_dir, f'../data/{self.dataset}/{self.valid_file}.txt'),
            'test': os.path.join(self.main_dir, f'../data/{self.dataset}/{self.test_file}.txt')
        }

# Set up parameters
params = Params(main_dir='/home/kip/Documents/assignments/drugs/KnowDDI/pytorch',  # Replace with your main directory
                dataset='drugbank',  # Replace with your dataset
                train_file='train',  # Replace with your train file
                valid_file='valid',  # Replace with your valid file
                BKG_file_name = 'BKG_file',
                add_traspose_rels=False,

                test_file='test')  # Replace with your test file

# Path to the .pth file
model_path = 'pth/best_graph_classifier.pth'

# Process the dataset
def data_processing(params):
    params.db_path = os.path.join(params.main_dir, os.path.abspath(f'../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}'))
    print("we are here")
    train_data = SubgraphDataset(db_path=params.db_path,
                                db_name='train_subgraph',
                                raw_data_paths=params.file_paths,
                                add_traspose_rels=params.add_traspose_rels,
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                dig_layer = params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)
    print("we are done")
    print(train_data)
data_processing(params)
#train_data, valid_data, test_data = process_dataset(params)
#print(train_data)
# Load the model
#model = torch.load(model_path, map_location=torch.device('cpu'))
#print(model.eval())