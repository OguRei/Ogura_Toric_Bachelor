import os
import torch
from torch.utils.data import Dataset
import toric_param
from toric_code_ogura import ToricCode
from tqdm import trange
from hyperparam import HyperParam as hp

class ToricCodeDataset(Dataset):
    def __init__(self, dataset_tensor_list, code_distance):
        self.dataset_tensor_list = dataset_tensor_list
        self.code_distance = code_distance

    def __len__(self):
        return len(self.dataset_tensor_list)

    def __getitem__(self, idx):
        data = self.dataset_tensor_list[idx]
        #data = torch.tensor([data])
        errors = torch.unsqueeze(data[:self.code_distance*2], dim=0) #[1, code_distance*2, code_distance]
        errors = torch.split(errors,self.code_distance,dim=1) #[1, code_distance, code_distance], [1, code_distance, code_distance]
        errors = torch.cat(errors, dim=0)
        errors_X = torch.unsqueeze(data[:self.code_distance*2], dim=0) #[1, code_distance*2, code_distance]
        errors_X = torch.split(errors_X,self.code_distance,dim=1) #[1, code_distance, code_distance], [1, code_distance, code_distance]
        errors_X = torch.cat(errors_X, dim=0)
        errors_Z = torch.unsqueeze(data[:self.code_distance*2], dim=0) #[1, code_distance*2, code_distance]
        errors_Z = torch.split(errors_Z,self.code_distance,dim=1) #[1, code_distance, code_distance], [1, code_distance, code_distance]
        errors_Z = torch.cat(errors_Z, dim=0)
        syndX = torch.unsqueeze(data[self.code_distance*2: self.code_distance*3], dim=0) #[1, code_distance, code_distance]
        syndZ = torch.unsqueeze(data[self.code_distance*3: self.code_distance*4], dim=0) #[1, code_distance, code_distance]
        return errors, errors_X, errors_Z, syndX, syndZ 

def save_dataset(dataset, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # ディレクトリが存在しない場合に作成
    torch.save({
        dataset[0],
    }, file_path)

def main(toric_code, num_data):
    # データセットを保存


    dataset_tensor_list = []
    for i in range(num_data):
        errors = toric_code.generate_errors()
        syndX = torch.from_numpy(toric_code.generate_syndrome_X(errors))
        syndZ = torch.from_numpy(toric_code.generate_syndrome_Z(errors))
        errors = torch.from_numpy(errors)
        errors_X = torch.where(((errors==0) | (errors==3)), 0, 1)
        errors_Z = torch.where(((errors == 0)| (errors==1)), 0, 1)
        errors = errors.to(torch.float32)
        syndX = syndX.to(torch.float32)
        syndZ = syndZ.to(torch.float32)
        errors_X = errors_X.to(torch.float32)
        errors_Z = errors_Z.to(torch.float32)
        data = torch.cat((errors, errors_X, errors_Z, syndX, syndZ),  dim  = 0)
        dataset_tensor_list.append(data)

    return dataset_tensor_list

    #print(dataset[0])
    #print(dataset[1])
"""
    print("-----------")
    print(dataset[2])
"""

if __name__ == "__main__":
    p=hp.error_rate #error rate
    size = hp.code_distance #code distance
    num_data = hp.num_data #データ数
    file_path = f"/home/mukailab/test/ogura_code/MyFiles/datasets/dataset_{num_data}_size{size}.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # ディレクトリが存在しない場合に作成

    toric_dataset_param = toric_param.toric_param(p, size)
    toric_code = ToricCode(toric_dataset_param)

    dataset_tensor_list = main(toric_code, num_data)
    torch.save(ToricCodeDataset(dataset_tensor_list, size), file_path)

"""
dataset = ToricCodeDataset(toric_dataset_param=toric_param.toric_param(p = 0.10,size = 5), num_data=1)
print(dataset.__getitem__)
"""

#main()
