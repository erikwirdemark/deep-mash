from sklearn.model_selection import train_test_split
from deep_mash.preprocess.gtzanstems_dataset import GTZANStemsDataset, ToLogMel
from torch.utils.data import DataLoader, Subset


# ---------- Split dataset by tracks ----------

def split_dataset_by_tracks(dataset: GTZANStemsDataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Extract unique track names
    all_tracks = sorted({p.name.split(".chunk")[0] for p in dataset.chunk_dirs})
    print(f"Total unique tracks: {len(all_tracks)}")
    val_relative = val_ratio / (train_ratio + val_ratio)
    
    temp_tracks, test_tracks = train_test_split(all_tracks, test_size=test_ratio, random_state=random_state)
    train_tracks, val_tracks = train_test_split(temp_tracks, test_size=val_relative, random_state=random_state)

    def get_chunk_indices(track_list):
        return [i for i, chunk_dir in enumerate(dataset.chunk_dirs) if chunk_dir.name.split(".chunk")[0] in track_list]

    train_dataset = Subset(dataset, get_chunk_indices(train_tracks))
    val_dataset = Subset(dataset, get_chunk_indices(val_tracks))
    test_dataset = Subset(dataset, get_chunk_indices(test_tracks))
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test chunks")

    return train_dataset, val_dataset, test_dataset

# ---------- DataLoaders ----------

def create_dataloaders(config, preprocess=False, num_workers=0):
    dataset = GTZANStemsDataset(config=config, preprocess=preprocess, preprocess_transform=ToLogMel(config=config))
    batch_size = config.training.batch_size

    train_dataset, val_dataset, test_dataset = split_dataset_by_tracks(dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def __main__():
    pass
# Skriva detta på något bättre sätt!!!