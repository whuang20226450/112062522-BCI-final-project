from ruamel.yaml import YAML
import numpy as np
import subprocess

if __name__ == "__main__":
  
  config_name = 'MnistEEG'
  config_name = 'LargeEEG'

  # weight = [float(w) for w in weight]
  # ratio = [34, 1.625, 5.79166667, 13.04166667, 3.20833333, 1.375, 3.375, 
  #          1, 1.08333333, 3.25, 1.66666667, 4.54166667, 2.58333333, 
  #          1.33333333, 1.29166667, 1.33333333, 5]
  # ratio = np.array(ratio)
  
  # v1 = [float(w) for w in 1 / np.sqrt(ratio)]
  # v2 = [float(w) for w in 1 / ratio]
  # v3 = [float(w) for w in 1 / ratio ** 2] 
  # min_threshold = 0.05
  # v4 = [max(_, min_threshold) for _ in v2]
  
  label_distribution = np.array([
    0.37792378, 0.01630435, 0.06836475, 0.1653653, 0.03938498, 0.01733627,
    0.04060608, 0.01188429, 0.01343217, 0.018265, 0.05577532, 0.01534122,
    0.01501445, 0.14500206])
  weight = 1 / (label_distribution / min(label_distribution))
  raweighttio = np.clip(weight, 0.05, 1)
  # weight[3] /= 2
  weight = [float(w) for w in weight]
  
  # for dropout in [0., 0.25, 0.5]:
  #   for init_lr in [1e-3, 1e-4, 1e-5]:
 

  for dataset_id in ['0000', '1100', '1001', '0101', '1101']:
    yaml = YAML()
    with open(f'config/{config_name}.yaml', 'r') as file:
      yaml_content = yaml.load(file)
      yaml_content["name"] = f"exp1_{config_name}_datasetId_{dataset_id}"
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