import torch
from torch.utils.data import DataLoader

class SquadDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, is_inference=False, shuffle=True):
        self.is_inference = is_inference
        super().__init__(dataset, collate_fn=self.squad_collate_fn, batch_size=batch_size, shuffle=shuffle)
        
    def squad_collate_fn(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        # return 6 tensors
        if self.is_inference:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_cls_index, all_p_mask, all_example_index
        # return 7 tensors
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_cls_index, all_p_mask, all_start_positions, all_end_positions



