import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import numpy as np

class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,51)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x

# Function to predict using a model
def predict_with_model(model_path, tokenizer, input_seq, input_mask):
    model = BERT_Arch(AutoModel.from_pretrained('bert-base-uncased'))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        preds = model(input_seq, input_mask)
        preds = preds.detach().cpu().numpy()

    return np.argmax(preds, axis=1)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Prepare unseen data
unseen_news_text = ["Your unseen news text here..."]
MAX_LENGTH = 200
tokens_unseen = tokenizer.batch_encode_plus(
    unseen_news_text,
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True
)

unseen_seq = torch.tensor(tokens_unseen['input_ids'])
unseen_mask = torch.tensor(tokens_unseen['attention_mask'])
subcategory_label_mapping = {'GOVTG': 0, 'DTOUR': 1, 'DODAF': 2, 'MOIAB': 3, 'GOVJH': 4, 'DLGLA': 5, 'MTRBL': 6, 'LGVED': 7, 
    'DOEAF': 8, 'DOSIR': 9, 'GOVCC': 10, 'GOVRJ': 11, 'AYUSH': 12, 'DSPRT': 13, 'MPOWR': 14, 'GOVTR': 15, 'DOLDR': 16, 
    'GOVPB': 17, 'DOFPD': 18, 'DOWCD': 19, 'DCLTR': 20, 'GOVKL': 21, 'MINPA': 22, 'DPHAM': 23, 'MINWR': 24, 'GOVJK': 25, 
    'GOVMP': 26, 'DHIND': 27, 'DHRES': 28, 'MOSPI': 29, 'MMSME': 30, 'DODIV': 31, 'GOVOR': 32, 'others': 33, 'GOVWB': 34, 
    'DMAFF': 35, 'DOARE': 36, 'DODWS': 37, 'DOFPI': 38, 'DOIPP': 39, 'MEAPM': 40, 'GOVAN': 41, 'GOVAP': 42, 'GOVGO': 43, 
    'GOVKN': 44, 'GOVUC': 45, 'CAGAO': 46, 'MCOAL': 47, 'MSHPG': 48, 'MONRE': 49, 'DOAHD': 50, 'GOVHP': 51, 'MMINE': 52, 
    'ARNPG': 53, 'DPUBE': 54, 'GOVCH': 55, 'GOVLK': 56, 'PRSEC': 57, 'DDPRO': 58, 'MTXTL': 59, 'ECCOM': 60, 'DATOM': 61, 
    'DOCND': 62, 'MOSTL': 63, 'GOVPY': 64, 'GOVDN': 65, 'MOYAS': 66}

category_label_mapping = {'MORLY': 0, 'GOVUP': 1, 'MOLBR': 2, 'MODEF': 3, 'MEAPD': 4, 'DORLD': 5, 'DOTEL': 6, 
    'others': 7, 'MINHA': 8, 'MOPRJ': 9, 'GOVAS': 10, 'DPLNG': 11, 'CBODT': 12, 'DARPG': 13, 'MINIT': 14, 'DHLTH': 15, 
    'DEABD': 16, 'DOSEL': 17, 'DOPAT': 18, 'GOVMH': 19, 'DDESW': 20, 'DOCAF': 21, 'DOCOM': 22, 'DEPOJ': 23, 'DPOST': 24, 
    'GOVTN': 25, 'MOEAF': 26, 'MORTH': 27, 'GNCTD': 28, 'GOVBH': 29, 'DOURD': 30, 'UIDAI': 31, 'DOAAC': 32, 'DSEHE': 33, 
    'DOSAT': 34, 'DEAID': 35, 'GOVHY': 36, 'DCOYA': 37, 'FADSS': 38, 'MPANG': 39, 'CBOEC': 40, 'MOSJE': 41, 'MOCOP': 42, 
    'MOCAV': 43, 'DEAPR': 44, 'GOVGJ': 45, 'DOEXP': 46, 'DOPPW': 47, 'DORVU': 48, 'MOMAF': 49, 'PMOPG': 50}

# Primary model prediction
primary_model_path = 'category_classifier.pt'
primary_preds = predict_with_model(primary_model_path, tokenizer, unseen_seq, unseen_mask)

# Check if primary prediction is label 7, indicating the need for secondary prediction
if primary_preds[0] == 7:
    secondary_model_path = 'subcategory_classifier.pt'
    secondary_preds = predict_with_model(secondary_model_path, tokenizer, unseen_seq, unseen_mask)
    # Use secondary_preds and subcategory_label_mapping for further processing/display
    #Label mapping:{'GOVTG': 0, 'DTOUR': 1, 'DODAF': 2, 'MOIAB': 3, 'GOVJH': 4, 'DLGLA': 5, 'MTRBL': 6, 'LGVED': 7, 
    # 'DOEAF': 8, 'DOSIR': 9, 'GOVCC': 10, 'GOVRJ': 11, 'AYUSH': 12, 'DSPRT': 13, 'MPOWR': 14, 'GOVTR': 15, 'DOLDR': 16, 
    # 'GOVPB': 17, 'DOFPD': 18, 'DOWCD': 19, 'DCLTR': 20, 'GOVKL': 21, 'MINPA': 22, 'DPHAM': 23, 'MINWR': 24, 'GOVJK': 25, 
    # 'GOVMP': 26, 'DHIND': 27, 'DHRES': 28, 'MOSPI': 29, 'MMSME': 30, 'DODIV': 31, 'GOVOR': 32, 'others': 33, 'GOVWB': 34, 
    # 'DMAFF': 35, 'DOARE': 36, 'DODWS': 37, 'DOFPI': 38, 'DOIPP': 39, 'MEAPM': 40, 'GOVAN': 41, 'GOVAP': 42, 'GOVGO': 43, 
    # 'GOVKN': 44, 'GOVUC': 45, 'CAGAO': 46, 'MCOAL': 47, 'MSHPG': 48, 'MONRE': 49, 'DOAHD': 50, 'GOVHP': 51, 'MMINE': 52, 
    # 'ARNPG': 53, 'DPUBE': 54, 'GOVCH': 55, 'GOVLK': 56, 'PRSEC': 57, 'DDPRO': 58, 'MTXTL': 59, 'ECCOM': 60, 'DATOM': 61, 
    # 'DOCND': 62, 'MOSTL': 63, 'GOVPY': 64, 'GOVDN': 65, 'MOYAS': 66}


    print(f"Secondary Prediction: {subcategory_label_mapping[secondary_preds[0]]}")
else:
    # Use primary_preds and category_label_mapping for display
    # Assuming category_label_mapping is defined, e.g.:
    # category_label_mapping = {'MORLY': 0, 'GOVUP': 1, 'MOLBR': 2, 'MODEF': 3, 'MEAPD': 4, 'DORLD': 5, 'DOTEL': 6, 
    #'others': 7, 'MINHA': 8, 'MOPRJ': 9, 'GOVAS': 10, 'DPLNG': 11, 'CBODT': 12, 'DARPG': 13, 'MINIT': 14, 'DHLTH': 15, 
    #'DEABD': 16, 'DOSEL': 17, 'DOPAT': 18, 'GOVMH': 19, 'DDESW': 20, 'DOCAF': 21, 'DOCOM': 22, 'DEPOJ': 23, 'DPOST': 24, 
    # 'GOVTN': 25, 'MOEAF': 26, 'MORTH': 27, 'GNCTD': 28, 'GOVBH': 29, 'DOURD': 30, 'UIDAI': 31, 'DOAAC': 32, 'DSEHE': 33, 
    # 'DOSAT': 34, 'DEAID': 35, 'GOVHY': 36, 'DCOYA': 37, 'FADSS': 38, 'MPANG': 39, 'CBOEC': 40, 'MOSJE': 41, 'MOCOP': 42, 
    # 'MOCAV': 43, 'DEAPR': 44, 'GOVGJ': 45, 'DOEXP': 46, 'DOPPW': 47, 'DORVU': 48, 'MOMAF': 49, 'PMOPG': 50}

    print(f"Primary Prediction: {category_label_mapping[primary_preds[0]]}")

