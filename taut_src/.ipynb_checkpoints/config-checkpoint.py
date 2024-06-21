import os

rootp = os.path.abspath(os.path.dirname(__file__))

smarts_path = os.path.join(rootp, "smarts_vmr.txt")

transform_path = os.path.join(rootp, "smirks_tansform_all.txt")

model_paths = [ 'constraint_training_output/data_v1/fold_1/models/best_model.pt',
                'constraint_training_output/data_v1/fold_3/models/best_model.pt' ,
                'constraint_training_output/data_v1/fold_4/models/best_model.pt' ]

model_paths = ["v10_data_training_siamese/fold_1/models/best_model.pt",
               "v10_data_training_siamese/fold_2/models/best_model.pt",
               "v10_data_training_siamese/fold_3/models/best_model.pt",
               "v10_data_training_siamese/fold_4/models/best_model.pt",
               "v10_data_training_siamese/fold_5/models/best_model.pt"]

model_paths = [os.path.join(rootp, "weights", path) for path in model_paths]
