import torch
import numpy as np

class CharacterPredictionLoss(torch.nn.Module):
  
    def __init__(self, char_to_id, device):
        super().__init__()

        self.char_to_id = char_to_id
        self.device = device
        self.loss_function = torch.nn.BCELoss()

    
    def forward(self, char_predictions, target_char):
        '''
        Calcualte CE loss
        We expect the index of target_char to have probability 1 
        '''
        
        target_index = target_char.flatten()
        targets = torch.zeros(char_predictions.shape, device=self.device)
        targets[torch.arange(targets.shape[0]), target_index, 0] = 1
        loss_mean = self.loss_function(char_predictions, targets)
        return loss_mean