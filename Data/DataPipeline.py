import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class DescriptorDataset(Dataset):
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], target_cols: List[str], problem_type: str = "regression"):
        self.data = data
        self.features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
        
        dtype = torch.float32 if problem_type == "regression" else torch.long
        self.targets = torch.tensor(data[target_cols].values, dtype=dtype)
        self.problem_type = problem_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataPipeline:
    @staticmethod
    def prepare_data(csv_path: str, feature_cols: List[str], target_cols: List[str], problem_type: str, batch_size: int = 32) -> DataLoader:
        """
        Robust, leakage-free data loader. Validation encoding and schema enforcement should happen here.
        """
        df = pd.read_csv(csv_path)
        
        # Verify schema
        missing_features = [c for c in feature_cols if c not in df.columns]
        missing_targets = [c for c in target_cols if c not in df.columns]
        
        if missing_features:
            raise ValueError(f"Missing feature columns in CSV: {missing_features}")
        if missing_targets:
            raise ValueError(f"Missing target columns in CSV: {missing_targets}")
            
        # Basic imputation
        df[feature_cols] = df[feature_cols].fillna(0.0)
        df[target_cols] = df[target_cols].fillna(0.0)
        
        dataset = DescriptorDataset(df, feature_cols, target_cols, problem_type)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
