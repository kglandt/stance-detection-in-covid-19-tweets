import torch
from torch import nn
from transformers import BertModel

class DAN_BERT(nn.Module):
    
    def __init__(self, hidden_dim, dropout, name):
        super(DAN_BERT, self).__init__()
        
        self.model_name = name
        
        self.F_obj = BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert', return_dict=True)
        self.F_subj = BertModel.from_pretrained('digitalepidemiologylab/covid-twitter-bert', return_dict=True)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
    
        # OBJECTIVE VIEW
        self.objective_domain_discriminator = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 1)
        )
        
        self.objective_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 2)
        )
        
        # SUBJECTIVE VIEW
        self.subjective_domain_discriminator = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 1)
        )
        
        self.subjective_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 2)
        )
        
        # FUSION Layer
        self.g = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.Sigmoid()
        )
        
        self.stance_classifier = nn.Sequential(
                                        nn.Linear(1024, self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.hidden_dim, 3)
        )
        
        
    def forward(self, input_ids, attention_mask, a=None):
        f_obj = self.F_obj(input_ids, attention_mask)['pooler_output']
        f_subj = self.F_subj(input_ids, attention_mask)['pooler_output']
        
        # FUSION
        f_obj_subj = torch.cat((f_obj, f_subj), dim=1)
        g = self.g(f_obj_subj)
        f_dual = (g * f_subj) + ((1 - g) * f_obj)
        stance_prediction = self.stance_classifier(f_dual)
        
        if a is not None:
            objective_prediction = self.objective_classifier(f_obj)
            subjective_prediction = self.subjective_classifier(f_subj)
            
            reverse_f_obj = f_obj
            objective_domain_prediction = self.objective_domain_discriminator(reverse_f_obj)
            
            reverse_f_subj = f_subj
            subjective_domain_prediction = self.subjective_domain_discriminator(reverse_f_subj)
            
            return stance_prediction, objective_prediction, subjective_prediction, objective_domain_prediction, subjective_domain_prediction
        
        return stance_prediction