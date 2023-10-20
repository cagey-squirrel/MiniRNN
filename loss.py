import torch
import numpy as np

class CharacterPredictionLoss(torch.nn.Module):
  
    def __init__(self, char_to_id, device, architecture):
        super().__init__()

        self.char_to_id = char_to_id
        self.device = device
        self.loss_function = torch.nn.BCELoss()
        self.architecture = architecture

    
    def forward(self, char_predictions_logits, targets):
        '''
        Calcualte CE loss
        We expect the index of target_char to have probability 1 
        '''
        
        char_predictions = torch.softmax(char_predictions_logits, dim=1)
        
        loss_mean = self.loss_function(char_predictions, targets)
        return loss_mean