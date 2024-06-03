from ruamel.yaml import YAML
import numpy as np
import subprocess

if __name__ == "__main__":
  
  config_name = 'LargeEEG'
    
  label_distribution = np.array([
    0.37792378, 0.01630435, 0.06836475, 0.1653653, 0.03938498, 0.01733627,
    0.04060608, 0.01188429, 0.01343217, 0.018265, 0.05577532, 0.01534122,
    0.01501445, 0.14500206])
  weight = 1 / (label_distribution / min(label_distribution))
  raweighttio = np.clip(weight, 0.05, 1)
  weight = [float(w) for w in weight]
 

  # for dataset_id in ['1111', '1121', '1151', '1131']:
  for dataset_id in ['1131']:
    yaml = YAML()
    with open(f'config/{config_name}.yaml', 'r') as file:
      yaml_content = yaml.load(file)
      yaml_content["name"] = f"exp2_{config_name}_datasetId_{dataset_id}"
      yaml_content["loss"]["weight"] = weight
      
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
        
    subprocess.run(f'python train.py start -c config/{config_name}.yaml', shell=True)     
    
      
  with open("experiments/result.txt", "a") as file:
    file.write(f"\n\n")