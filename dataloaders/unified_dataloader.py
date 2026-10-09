import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from dataloaders.config import DATASET_CONFIG

class UnifiedDataLoader:
    def __init__(self, config, batch_size=2, temporal_length=2, num_workers_per_dataset=8, shuffle=True, pin_memory=True):
        """Initialize MixedDataLoader with a task/dataset config structure.
        
        Args:
            config (dict): Dict of tasks with their weights and datasets
                Format:
                {
                    'task1': {
                        'task_weight': float,
                        'dataset': [dataset1, dataset2, ...],
                        'dataset_weights': [weight1, weight2, ...]
                    },
                    'task2': {...}
                }
            batch_size (int): Batch size for the data loader
            temporal_length (int): temporal length for video datasets or frame packing for image datasets
            num_workers_per_dataset (int): Number of workers for each data loader
            shuffle (bool): Whether to shuffle the data loader
            pin_memory (bool): Whether to pin memory for the data loader, if True, data transfer to GPU is faster

        Note: 
            - This dataloader supports both image and video datasets, and you can mix them together.
            - For image datasets, the dataset name should start with "Image"
            - For video datasets, the dataset name should start with "Video"

        Example config:
            - Image datasets:
                {
                    'depth': {
                        'task_weight': 1.0,
                        'dataset': ['ImageDepthHypersim', 'ImageDepthPiat(dataset="sniklaus-side_undepth", resize=400)'],
                        'dataset_weights': [1.0, 1.0]
                    },
                    'normals': {
                        'task_weight': 1.0,
                        'dataset': ['ImageNormalHypersim', 'ImageNormalVirtualKITTI2'],
                        'dataset_weights': [1.0, 1.0]
                    },
                }

            - Video datasets:
                {
                    'depth': {
                        'task_weight': 1.0,
                        'dataset': ['VideoDepthPointOdyssey(stride_range=(1, 3))', 'VideoDepthTartanair(stride_range=(1, 3))'],
                        'dataset_weights': [1.0, 1.0]
                    },
                }
        """

        self.config = config
        self.batch_size = batch_size
        self.temp_length = temporal_length
        self.num_workers_per_dataset = num_workers_per_dataset
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        # Get list of all tasks
        self.tasks = list(config.keys())
        
        # Get task weights
        self.task_weights = [config[task]['task_weight'] for task in self.tasks]
        
        # Initialize all dataloaders for each task and dataset
        self.task_datasets = {}
        self.dataloaders = {}
        
        for task in self.tasks:
            task_config = config[task]
            
            # Get dataset names and weights for this task
            dataset_names = task_config['dataset']
            dataset_weights = task_config.get('dataset_weights', [1.0] * len(dataset_names))
            
            # Ensure dataset names and weights match in length
            if len(dataset_names) != len(dataset_weights):
                raise ValueError(f"Number of datasets and weights must match for task {task}")
            
            # Initialize datasets for this task
            self.task_datasets[task] = {
                'datasets': [],
                'dataset_names': [],
                'weights': dataset_weights,
                'dataloaders': []
            }

            for ds_name in dataset_names:
                # Import and create dataset instance based on dataset name
                try:
                    # Check if dataset name contains parameters
                    is_video_dataset = ds_name.startswith('Video')
                    import_dir = 'dataloaders.datasets.video_datasets' if is_video_dataset else 'dataloaders.datasets.image_datasets'
                    
                    # Parse dataset name and parameters
                    if '(' in ds_name and ')' in ds_name:
                        # Parse the base name and parameters
                        base_name = ds_name.split('(')[0]
                        params_str = ds_name.split('(', 1)[1].rsplit(')', 1)[0].strip()
                        
                        # Import dataset class
                        dataset_module = __import__(import_dir, fromlist=[base_name])
                        dataset_class = getattr(dataset_module, base_name)
                        
                        # Check if parameters are empty
                        if params_str:
                            # Evaluate the parameters string to get kwargs
                            kwargs = eval(f"dict({params_str})")
                            # Create dataset with parameters
                            if is_video_dataset:
                                assert "T" not in kwargs, "Parameter 'T' is reserved for self.temp_length and should not be used in dataset parameters."
                                kwargs['T'] = self.temp_length
                            dataset = dataset_class(**kwargs)
                        else:
                            # Empty parentheses, create dataset without parameters
                            dataset = dataset_class()
                    else:
                        # No parentheses, import and create dataset without parameters
                        dataset_module = __import__(import_dir, fromlist=[ds_name])
                        dataset_class = getattr(dataset_module, ds_name)
                        dataset = dataset_class()
                    
                    # Create DataLoader for this dataset
                    base_ds_name = base_name if '(' in ds_name else ds_name
                    batch_size = self.batch_size if is_video_dataset else self.batch_size * self.temp_length    # frame packing for image datasets
                    if "Piat" in base_ds_name:
                        dataloader = dataset.loader(batch_size=batch_size, num_workers=self.num_workers_per_dataset, shuffle=self.shuffle)
                    else:   
                        dataloader = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers_per_dataset,
                            drop_last=True,
                            pin_memory=self.pin_memory,
                        )
                    
                    # Store dataset and its dataloader
                    self.task_datasets[task]['datasets'].append(dataset)
                    self.task_datasets[task]['dataset_names'].append(ds_name)
                    self.task_datasets[task]['dataloaders'].append(dataloader)
                    
                except (ImportError, AttributeError):
                    raise ValueError(f"Could not find dataset: {ds_name}")
        # Dictionary to store iterators for each dataloader
        # Format: {(task, dataset_idx): iterator}
        self.iterators = {}
        
        # Initialize all iterators
        for task in self.tasks:
            selected_task_datasets = self.task_datasets[task]
            for dataset_idx in range(len(selected_task_datasets['datasets'])):
                key = (task, dataset_idx)
                dataset_name = selected_task_datasets['dataset_names'][dataset_idx]
                
                if 'Piat' in dataset_name:
                    self.iterators[key] = selected_task_datasets['dataloaders'][dataset_idx]
                else:
                    self.iterators[key] = iter(selected_task_datasets['dataloaders'][dataset_idx])

    def __iter__(self):
        # Just return self without selecting a specific task/dataset
        return self
    
    def __next__(self):

        while True:
            # Sample a task based on task weights
            current_task = random.choices(
                self.tasks, 
                weights=self.task_weights, 
                k=1
            )[0]
            
            # Get task info
            selected_task_datasets = self.task_datasets[current_task]
            
            # Sample a dataset based on dataset weights
            dataset_idx = random.choices(
                range(len(selected_task_datasets['datasets'])), 
                weights=selected_task_datasets['weights'], 
                k=1
            )[0]
            current_dataset_name = selected_task_datasets['dataset_names'][dataset_idx]
            
            # Create a key for this task-dataset combination
            key = (current_task, dataset_idx)
            
            # Try to get a batch from the current loader
            try:
                try:
                    # Try to get the next batch from the existing iterator
                    batch = next(self.iterators[key])
                except StopIteration:
                    # If iterator is exhausted, create a new one
                    self.iterators[key] = iter(selected_task_datasets['dataloaders'][dataset_idx])
                    batch = next(self.iterators[key])
                
                # Skip if batch is None or empty
                if batch is None or not batch:
                    continue
                
                # Check valid flag in batch
                if 'valid' in batch:
                    if isinstance(batch['valid'], (list, torch.Tensor)):
                        # Multi-sample batch: skip if any sample is invalid
                        if isinstance(batch['valid'], torch.Tensor):
                            if not torch.all(batch['valid']):
                                continue
                        else:
                            if not all(batch['valid']):
                                continue
                    else:
                        # Single-sample batch
                        if not batch['valid']:
                            continue

                # Add task/dataset information to the batch
                batch['task'] = current_task
                batch['dataset'] = current_dataset_name
                batch['is_video'] = current_dataset_name.startswith('Video')

                # For image datasets, we need to handle frame packing
                if current_dataset_name.startswith('Image'):
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            sh = v.shape
                            batch[k] = v.view(-1, self.temp_length, *sh[1:])  # (real bs, T, C, H, W)
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if torch.is_tensor(vv):
                                    sh = vv.shape
                                    batch[k][kk] = vv.view(-1, self.temp_length, *sh[1:])  # (real bs, T, C, H, W)
                    
                return batch
            except StopIteration:
                # This should only happen if a dataloader is completely empty
                raise StopIteration
            
if __name__ == "__main__":
    import imageio, logging, os
    logging.basicConfig(level=logging.INFO)
    config = {
        'geometric_image': {
            'task_weight': 1.0,
            'dataset': ['ImageHypersimDataset()', 'ImageVKITTI2Dataset()'],
            'dataset_weights': [1.0, 1.0]
        },
        'semantic_image': {
            'task_weight': 1.0,
            'dataset': ['ImageCOCODataset()', 'ImageCityscapesDataset()'],
            'dataset_weights': [1.0, 1.0]
        },
        'geometric_video': {
            'task_weight': 1.0,
            'dataset': [
                'VideoPointOdysseyDataset(max_frame_skip=8)', 
                'VideoTartanairDataset(max_frame_skip=8)',
                'VideoSpringDataset(max_frame_skip=8)'
            ],
            'dataset_weights': [1.0, 1.0, 1.0]
        },
        'semantic_video': {
            'task_weight': 1.0,
            'dataset': [
                'VideoVIPSegDataset(max_frame_skip=4)',
            ],
            'dataset_weights': [1.0]
        },
    }
    temporal_length = 4
    torch.manual_seed(1); rand_proj = torch.rand(3,17)

    mixed_dataloader = UnifiedDataLoader(config, batch_size=1, temporal_length=temporal_length, num_workers_per_dataset=0, shuffle=True, pin_memory=True)
    
    for i, batch in enumerate(mixed_dataloader):
        task_datasets = mixed_dataloader.task_datasets[batch['task']]
        dataset = task_datasets['datasets'][task_datasets['dataset_names'].index(batch['dataset'])]
        x = []
        b = 0  # only first batch element
        for t in range(temporal_length):
            rgb = ((batch['rgb'][b,t] + 1)/2).permute(1,2,0).numpy()
            depth = batch['depth'][b,t].permute(1,2,0).numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth = np.concatenate([depth]*3, axis=2)
            normal = ((batch['normal'][b,t] + 1)/2).permute(1,2,0).numpy()
            semseg_coco = batch['semseg-coco'][b,t]
            semantic_ids = torch.unique(semseg_coco)
            semseg_vis = torch.zeros(rgb.shape)
            for sid in semantic_ids:
                semseg_vis[semseg_coco==sid] = dataset.sem_color_code[sid.long()]
            semseg_coco = semseg_vis.numpy()

            semseg_cs = batch['semseg-cityscapes'][b,t]
            semantic_ids = torch.unique(semseg_cs)
            semseg_vis = torch.zeros(rgb.shape)
            for sid in semantic_ids:
                semseg_vis[semseg_cs==sid] = dataset.sem_color_code[sid.long()]
            semseg_cs = semseg_vis.numpy()

            semseg_vip = batch['semseg-vipseg'][b,t]
            semantic_ids = torch.unique(semseg_vip)
            semseg_vis = torch.zeros(rgb.shape)
            for sid in semantic_ids:
                semseg_vis[semseg_vip==sid] = dataset.sem_color_code[sid.long()]
            semseg_vip = semseg_vis.numpy()

            boundary = batch['boundary'][b,t].permute(1,2,0).numpy()
            boundary = np.concatenate([boundary]*3, axis=2)

            keypoint = rand_proj @ batch['keypoint-coco'][b,t].reshape(17, -1)  # 17, H, W
            keypoint = keypoint.reshape(3, *rgb.shape[:2]).permute(1,2,0).numpy()
            keypoint = (keypoint - keypoint.min()) / (keypoint.max() - keypoint.min() + 1e-8)

            out_img = np.hstack((rgb, depth, normal, semseg_coco, semseg_cs, semseg_vip, boundary, keypoint)) * 255
            x.append(out_img)
        os.makedirs("data_vis", exist_ok=True)
        imageio.imwrite(f"data_vis/uni_loader_sample_{i}.png", np.vstack(x).astype(np.uint8))
        print(dataset, batch['supervision']['rgb'].shape, batch['rgb'].shape)
        
        if i == 9:
            break
    del mixed_dataloader