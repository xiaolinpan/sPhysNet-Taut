import os

rootp = os.path.abspath(os.path.dirname(__file__))

smarts_path = os.path.join(rootp, "smarts_vmr.txt")

transform_path = os.path.join(rootp, "smirks_tansform_all.txt")

model_paths = [ 'constraint_training_output/best_model1.pt',
                'constraint_training_output/best_model3.pt' ,
                'constraint_training_output/best_model4.pt' ]

#model_paths = ["v10_data_training_siamese/best_model1.pt",
#               "v10_data_training_siamese/best_model2.pt",
#               "v10_data_training_siamese/best_model3.pt",
#               "v10_data_training_siamese/best_model4.pt",
#               "v10_data_training_siamese/best_model5.pt"]

model_paths = ["v17-data-v12/best_model1.pt",
               "v17-data-v12/best_model2.pt",
               "v17-data-v12/best_model3.pt",
               "v17-data-v12/best_model4.pt",
               "v17-data-v12/best_model5.pt"]

model_paths = [os.path.join(rootp, "weights", path) for path in model_paths]
