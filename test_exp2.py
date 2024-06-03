from ruamel.yaml import YAML
import numpy as np
import subprocess

if __name__ == "__main__":
  
  config_name = 'LargeEEG' 

  for dataset_id in ['1111', '1121', '1151', '1131']:
    yaml = YAML()
    with open(f'config/{config_name}.yaml', 'r') as file:
      yaml_content = yaml.load(file)
      yaml_content["name"] = f"exp2_{config_name}_datasetId_{dataset_id}_test"
      yaml_content["weight_path"] = f"experiments/exp2_{config_name}_datasetId_{dataset_id}/weights/model_best_macrof1.pt"
      
      if dataset_id[-1] == '1':
        yaml_content["dataset"]["preprocess"] = True
      else:
        yaml_content["dataset"]["preprocess"] = False
        
      if dataset_id[0] == '1':
        yaml_content["train_setting"]["num_channels"] = 17
      else: 
        yaml_content["train_setting"]["num_channels"] = 63
              
      yaml_content["dataset"]["dataset_id"] = dataset_id
      
      with open(f'config/{config_name}.yaml', 'w') as file:
        yaml.dump(yaml_content, file)
        
    subprocess.run(f'python train.py test -c config/{config_name}.yaml', shell=True)   
    
      
  with open("experiments/result.txt", "a") as file:
    file.write(f"\n\n")