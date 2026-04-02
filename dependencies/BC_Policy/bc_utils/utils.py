import numpy as np
import enum, os, json

class LossType(enum.Enum):
    MSE = "mse"
    LOG_PROB = "log"
    
    
def get_data_stata(data):
    """
        compute the data stata for each dimension: mean and std
        assume the data shape is (N, D)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

def normalize(data, mean, std, epsilon=1e-8):
    std = np.array(std)
    safe_stds = np.where(std < epsilon, epsilon, std)
    normalized_data = (data - mean) / safe_stds
    return normalized_data

def denormalize(data, mean, std, epsilon=1e-8):
    std = np.array(std)
    safe_stds = np.where(std < epsilon, epsilon, std)
    original_data = data * safe_stds + mean
    return original_data

def save_data_stat(stata_dict, saving_path):
    save_dir = os.path.dirname(saving_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"created {save_dir} for saving data stata")
        
    for key, data in stata_dict.items():
        if not isinstance(data, list):
            stata_dict[key] = data.tolist()
    
    with open(saving_path, "w") as f:
        json.dump(stata_dict, f, indent=4)
        
def load_data_stat(path):
    if not os.path.exists(path):
        raise ValueError(f'data stat {path} does not exist')
    
    with open(path, "r") as f:
        stata_dict = json.load(f)
    return stata_dict
