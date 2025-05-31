#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nemo.collections.asr as nemo_asr
import pandas as pd
from tqdm.auto import tqdm
import jiwer
import torch
import torchaudio
import copy

from trucles_model import *
from trucles_utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import sentencepiece as spm

import yaml
import argparse


# In[2]:
parser = argparse.ArgumentParser(description='TruCLeS training')
parser.add_argument('--config', required=True, dest='config_file',
                    help='Requires full path to config file')

args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


DEVICE= config['DEVICE']
asr_checkpoint_path = config['asr_checkpoint_path']
TRAIN_BS= config['TRAIN_BS']
TEST_BS= config['TEST_BS']
DEV_BS = config['DEV_BS']
EPOCHS= config['EPOCHS']

confid_model_name= config['confid_model']
input_dim= config['input_dim']
hidden_dim= config['hidden_dim']
learning_rate= config['learning_rate']
trucles_train_dataset= config['trucles_train_dataset']
trucles_dev_dataset= config['trucles_dev_dataset']
best_loss_checkpoint= config['best_loss_checkpoint']
checkpoint= config['checkpoint']
loss= config['loss']
audio_root= config['audio_root']
spm_model_path=config['spm_model_path']
a=config['shrinkage_loss_a']
c=config['shrinkage_loss_c']


# In[3]:


asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=asr_checkpoint_path)


# In[4]:


asr_model.freeze() # inference mode
asr_model = asr_model.to(DEVICE) # transfer model to device

blank = len(asr_model.decoder.vocabulary)
unkwn = 0  
vocab = copy.deepcopy(asr_model.decoder.vocabulary)
print(vocab)
vocab.append('@')
print(vocab)
print(len(vocab))

sp = spm.SentencePieceProcessor()
sp.load(spm_model_path)


# In[5]:


if confid_model_name=='linear':
    confid_model = trucles_linear(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
if confid_model_name=='linear_deep':
    confid_model = trucles_linear_new(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
if confid_model_name=='lstm':
    confid_model = trucles_lstm(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)


print(confid_model)


# In[6]:


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, a, c):
        l2criterion = nn.MSELoss(reduction='mean')
        l2loss = l2criterion(output, target)
        l1criterion = nn.L1Loss(reduction='mean')
        l1loss = l1criterion(output, target)
        shrinkage = (1 + (a*(c-l1loss)).exp()).reciprocal()
        loss = shrinkage * l2loss * output.exp()
        loss = torch.mean(loss)
        
        return loss


# In[7]:


if loss == 'mae':
    criterion_confid = nn.L1Loss(reduction='mean').to(DEVICE)
if loss == 'shrinkage':
    criterion_confid = CustomLoss().to(DEVICE)

optimizer_confid = torch.optim.Adam(confid_model.parameters(), lr=learning_rate, 
                       weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)


# In[8]:


class truclesDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """
    def __init__(self, data_frame, audio_root):
        self.data_frame = data_frame
        self.audio_root = audio_root

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        audio_file = self.audio_root + self.data_frame.iloc[idx, 0]
        input_signal, sr = torchaudio.load(audio_file, normalize = True)
        channel_dim = input_signal.shape[0]
        if channel_dim > 1:
            waveform = torch.mean(waveform, 0, keepdim=True)
        if sr!=16000:
            resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
            input_signal = resampler(input_signal)
        input_signal_length = torch.tensor([input_signal.shape[-1]], dtype=torch.long)
         
        transcript = self.data_frame.iloc[idx,1]
        transcript = str(transcript)
       
        return input_signal, input_signal_length, transcript
    
def collate_batch_trucles(batch):
    
    input_signal_temp_list, input_signal_length_list, references = [], [], []

    for (_input_signal, _input_signal_length, _transcript) in batch:
        input_signal_temp_list.append(_input_signal.squeeze())
        input_signal_length_list.append(_input_signal_length)
        references.append(_transcript)
        
    input_signal_list = nn.utils.rnn.pad_sequence(input_signal_temp_list, batch_first=True)    
    return input_signal_list, input_signal_length_list, references


# In[12]:


df_train = pd.read_csv(trucles_train_dataset, sep='\t', header=None)

df_valid = pd.read_csv(trucles_dev_dataset, sep='\t', header=None)
print(df_train)
print(df_valid)

df_train = df_train.sample(frac=1).reset_index(drop=True)
df_valid = df_valid.sample(frac=1).reset_index(drop=True)

trucles_trainset = truclesDataset(df_train, audio_root)

trucles_trainloader = torch.utils.data.DataLoader(trucles_trainset, batch_size = TRAIN_BS, shuffle = True, 
                                          collate_fn = collate_batch_trucles, drop_last=True, 
                                          num_workers=4, pin_memory=True)

trucles_devset = truclesDataset(df_valid, audio_root)

trucles_devloader = torch.utils.data.DataLoader(trucles_devset, batch_size = DEV_BS, shuffle = True, 
                                          collate_fn = collate_batch_trucles, drop_last=True, 
                                          num_workers=4, pin_memory=True)


# In[13]:



# In[14]:


def levenshtein_distance(str1, str2):
    # Initialize a matrix to store the distances
    distance_matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Initialize the first row and column
    for i in range(len(str1) + 1):
        distance_matrix[i][0] = i
    for j in range(len(str2) + 1):
        distance_matrix[0][j] = j

    # Fill in the rest of the matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1,  # deletion
                                         distance_matrix[i][j - 1] + 1,  # insertion
                                         distance_matrix[i - 1][j - 1] + cost)  # substitution

    # Return the bottom-right cell of the matrix
    return distance_matrix[len(str1)][len(str2)]


# In[15]:


def fetch_word_ops(REF, HYP, TRAIN_BS):
    lines = jiwer.visualize_alignment(jiwer.process_words(REF, HYP), show_measures=False, skip_correct=False).split('\n')
    word_ops_list = []
    ref_j_list = []
    hyp_j_list = []
    j = 1
    k = 2
    
    for i in range(TRAIN_BS):
        ref_align = lines[j+(5*i)][5:]
        hyp_align = lines[k+(5*i)][5:]
        
        word_ops = []
        
        ref_j = ref_align.split()
        hyp_j = hyp_align.split()
        ref_j_list.append(ref_j)
        hyp_j_list.append(hyp_j)
        
        for i in range(len(ref_j)):
            if ref_j[i] == hyp_j[i]:
                word_ops.append('C')
            elif '*' in ref_j[i]:
                word_ops.append('I')
            elif '*' in hyp_j[i]:
                word_ops.append('D')
            else:
                word_ops.append('S')
        word_ops_list.append(word_ops)
   
    return word_ops_list, ref_j_list, hyp_j_list


# In[16]:


def find_pred_ranges(with_blank_ids_list, HYP, vocab, uncollapsed_HYP):
    # print(with_blank_ids_list)
    words_list = []
    for i in range(len(HYP)):
        words = HYP[i].split()
        words_list.append(words)
    # print(words_list)
    temp_hyp_list = []

    for j in range(len(with_blank_ids_list)):   
        temp_hyp = []
        for i in range(len(with_blank_ids_list[j])):
            temp_hyp.append((vocab[with_blank_ids_list[j][i]], i))
        temp_hyp_list.append(temp_hyp)   
    # print(temp_hyp_list)

    # New list to store the result
    col_hyp_list = []
    
    for j in range(len(temp_hyp_list)):
        col_hyp = []
        # Iterate over the original list
        for l in range(len(temp_hyp_list[j])):
            # If it's the first element or the current element is different from the previous one
            if l == 0 or temp_hyp_list[j][l][0] != temp_hyp_list[j][l - 1][0]:
                # Append the current element to the result list
                col_hyp.append(temp_hyp_list[j][l])
        col_hyp_list.append(col_hyp)

    final_hyp_list = []
    for j in range(len(col_hyp_list)):   
        temp_hyp = []
        for i in range(len(col_hyp_list[j])):
            if col_hyp_list[j][i][0] != '@':
                temp_hyp.append(col_hyp_list[j][i])
        final_hyp_list.append(temp_hyp)   

    word_offset_list = []
    
    for j in range(len(with_blank_ids_list)):
        words = words_list[j]
        # print(words)
        k = 0
        one_word = words[k]

        tokens = []
        iter = 0
        start_offset = 0
        word_offset = []
        while k<=len(words)-1:
            tokens.append(final_hyp_list[j][iter][0])
            # print(tokens)
            if one_word == asr_model.decoding.decode_tokens_to_str(tokens):

                ext = True
                temp_toks = list(tokens)
                temp_iter = iter
                while ext==True:
                    temp_iter+=1
                    if temp_iter == len(final_hyp_list[j]):
                        break
                    temp_toks.append(final_hyp_list[j][temp_iter][0])
                    if one_word == asr_model.decoding.decode_tokens_to_str(temp_toks):
                        ext = True
                    else:
                        ext = False
                        iter = temp_iter-1
                end_offset = final_hyp_list[j][iter][1]

                pipe_word = '|'.join(uncollapsed_HYP[j][start_offset:end_offset+1])
                word_offset.append((pipe_word, start_offset, end_offset))
                start_offset = end_offset + 1
                k = k + 1
                if k == len(words):
                    break
                else:
                    one_word = words[k]
                tokens = []
            iter+=1
            if iter == len(final_hyp_list[j]):
                break

        word_offset_list.append(word_offset)
    return word_offset_list


# In[17]:


def fetch_word_ops_single(REF, HYP):
    lines = jiwer.visualize_alignment(jiwer.process_words(REF, HYP), show_measures=False, skip_correct=False).split('\n')

    ref_align = lines[1][5:]
    hyp_align = lines[2][5:]
    
    word_ops = []
    
    ref_j = ref_align.split()
    hyp_j = hyp_align.split()
    
    for i in range(len(ref_j)):
        if ref_j[i] == hyp_j[i]:
            word_ops.append('C')
        elif '*' in ref_j[i]:
            word_ops.append('I')
        elif '*' in hyp_j[i]:
            word_ops.append('D')
        else:
            word_ops.append('S')


    return ref_j, hyp_j, word_ops


# In[18]:


def actual_score_gen(references, HYP, uncollapsed_HYP, word_ops_list, ref_j_list, hyp_j_list, word_offset_list_temp, soft):
      
    del_positions_list = []
    for i in range(len(word_ops_list)):
        del_positions = [index for index, char in enumerate(word_ops_list[i]) if char == 'D']
        del_positions_list.append(del_positions)
          
    tuple_to_insert = ('$', '$', '$')
    # Insert the tuple at each specified position
    for i in range(len(del_positions_list)):
        for pos in del_positions_list[i]:
            word_offset_list_temp[i].insert(pos, tuple_to_insert)
        
    word_score_list = []

    for m in range(len(hyp_j_list)):
        
        soft_temp = soft[m]
        soft_temp = soft_temp.unsqueeze(0)
        word_score = []
    
        for i in range(len(hyp_j_list[m])):
            if word_ops_list[m][i] == 'D':
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i])
                continue
            if word_ops_list[m][i] == 'I':
                score = 0
                # print(ref_j[i], ',', hyp_j[i], ',', word_ops[i], ',', score)
                word_score.append(score)
        
            if word_ops_list[m][i] == 'S':
                score = 0
                weight = 0
                start_offset = word_offset_list_temp[m][i][1]
                end_offset = word_offset_list_temp[m][i][2]
                ref_toks = sp.encode_as_pieces(ref_j_list[m][i])
               
                word = word_offset_list_temp[m][i][0]
                token_list = word.split('|')
        
                # Create a list with numbers in the specified range
                number_list = [num for num in range(start_offset, end_offset + 1)]
                
                indices_list = []
                for j in range(len(number_list)):
                    if token_list[j]=='@':
                        continue
                    else:
                        indices_list.append(number_list[j])
        
                char_to_remove = '@'
                hyp_toks = [string for string in token_list if string != char_to_remove]
                # print(hyp_toks)
                
                merged_ref = " ".join(ref_toks)
                merged_hyp = " ".join(hyp_toks)
        
                ref_subs_j, hyp_subs_j, word_subs_ops = fetch_word_ops_single(merged_ref, merged_hyp)   
                
                del_count = 0
                for k in range(len(hyp_subs_j)):
                    if word_subs_ops[k]=='D':
                        del_count+=1
                        continue
                        
                    if word_subs_ops[k]=='I':
                        weight+=1
                        score+=0
                        
        
                    if word_subs_ops[k]=='C':
                        weight+=1
                        score+=soft_temp[:,indices_list[k-del_count]].max(dim=-1, keepdim=False).values.item()
                        
                        
                    if word_subs_ops[k]=='S':
                        true_token_index = vocab.index(ref_subs_j[k])
                        tcp = soft_temp[:,indices_list[k-del_count]].view(-1)[true_token_index].item()
                        
                        weight+=1
                        score+=tcp
                        
                # Calculate Levenshtein distance
                lev_distance = levenshtein_distance(''.join(ref_toks), ''.join(hyp_toks))
                
                # Calculate normalized Levenshtein distance
                max_len = max(len(''.join(ref_toks)), len(''.join(hyp_toks)))
                normalized_lev_distance = lev_distance / max_len
                
                
                score = (score/weight)*(1-normalized_lev_distance)
                word_score.append(score)
                
        
            if word_ops_list[m][i] == 'C':
                score = 0
        
                start_offset = word_offset_list_temp[m][i][1]
                weight = 0
                token_list = word_offset_list_temp[m][i][0].split('|')
                for l in range(len(token_list)):
                    if token_list[l] == '@':
                        continue
                    else:
                        weight+=1 
                        score+=soft_temp[:,l+start_offset].max(dim=-1, keepdim=False).values.item()
                score = score/weight
                word_score.append(score)
                
        word_score_list.append(word_score)
    return word_score_list


# In[19]:


def ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab):
    # vocab.append('@')
    encoder_output = asr_model.encoder(audio_signal=processed_signal_list, length=processed_signal_length_list)
    encoded = encoder_output[0]
    
    if asr_model.decoder.is_adapter_available():
        encoded = encoded.transpose(1, 2)  # [B, T, C]
        encoded = asr_model.forward_enabled_adapters(encoded)
        encoded = encoded.transpose(1, 2)  # [B, C, T]
    
    decoder_out = asr_model.decoder.decoder_layers(encoded)
   
    if asr_model.decoder.temperature != 1.0:
        soft = (decoder_out.transpose(1, 2)/asr_model.decoder.temperature).softmax(dim=-1)
    else:
        soft = decoder_out.transpose(1, 2).softmax(dim=-1)

    greedy_predictions = soft.argmax(dim=-1, keepdim=False)
    
    HYP = []
    uncollapsed_HYP = []
    with_blank_ids_list = []
    for i in range(greedy_predictions.shape[0]):
        with_blank_ids = [label for label in greedy_predictions[i].tolist()]
       
        greedy_predictions1 = torch.unique_consecutive(greedy_predictions[i], dim=-1)
        non_blank_ids = [label for label in greedy_predictions1.tolist() if label!=blank]
        non_blank_ids = [label for label in non_blank_ids if label!=unkwn] #added aug 31 24
        tokens = asr_model.decoding.decode_ids_to_tokens(non_blank_ids)
        HYP.append(asr_model.decoding.decode_tokens_to_str(tokens))
        
        with_blank_hyp = []
        for i in range(len(with_blank_ids)):
            with_blank_hyp.append(vocab[with_blank_ids[i]])
        
        uncollapsed_HYP.append(with_blank_hyp)
        with_blank_ids_list.append(with_blank_ids)

    return encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list


# In[20]:


def train(confid_model, optimizer_confid, criterion_confid, batch, asr_model, device, TRAIN_BS, vocab):
    confid_model.train()
    optimizer_confid.zero_grad()
    
    input_signal_list, input_signal_length_list, references = batch
    processed_signal_list, processed_signal_length_list = asr_model.preprocessor(
                    input_signal=input_signal_list.to(DEVICE), length=torch.tensor(input_signal_length_list, dtype=torch.float32).to(DEVICE),
                ) 
    encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list = ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab) 
    
    for i in range(len(HYP)):
        if HYP[i] == '':
            return 0
   
    word_ops_list, ref_j_list, hyp_j_list = fetch_word_ops(references, HYP, TRAIN_BS)
   
    word_offset_list = find_pred_ranges(with_blank_ids_list, HYP, vocab, uncollapsed_HYP)
    
    word_offset_list_temp = copy.deepcopy(word_offset_list) 
    
    try:
        actual_score_list = actual_score_gen(references, HYP, uncollapsed_HYP, word_ops_list, ref_j_list, hyp_j_list, word_offset_list_temp, soft)
    except:
        print("REF:", references)
        print("HYP:", HYP)
        print("OPS:", word_ops_list)
        print("OFFSET:", word_offset_list)
        return 0


    soft = soft.permute(0, 2, 1)
    
    
    confid_tensor = []
    final_loss = 0
    input_stack = []
 
    for i in range(len(word_offset_list)):
        for j in range(len(word_offset_list[i])):
            start_frame = word_offset_list[i][j][1]
            
            end_frame = word_offset_list[i][j][2]
            indices = [*range(start_frame, end_frame+1, 1)]
            
            encoded_input = encoded[i,:,indices]
            decoder_input = decoder_out[i,:,indices]
            soft_input = soft[i,:,indices]
            
            encoded_input = torch.mean(encoded_input, 1, keepdim=False) #If false, shape is [256]. If True, shape is [1,256]
            decoder_input = torch.mean(decoder_input, 1, keepdim=False)
            soft_input = torch.mean(soft_input, 1, keepdim=False)
            
            confid_input = [encoded_input, decoder_input, soft_input]
            confid_tensor = torch.cat(confid_input, dim=-1)
            
            input_stack.append(confid_tensor)
    
    input_stack = torch.stack(input_stack)
    score = confid_model(input_stack)
    actual_score = [] 
    for sublist in actual_score_list:
        for item in sublist:
            actual_score.append(item)
    target_score = Variable(torch.FloatTensor(actual_score), requires_grad = True).to(device)

    if loss == 'mae':
        try:
            loss_confid = criterion_confid(score.squeeze(dim=-1), target_score)
        except Exception as e:
            print(e)
                        
    if loss == 'shrinkage':
        loss_confid = criterion_confid(score.squeeze(dim=-1), target_score, a, c)
   
    loss_confid.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(confid_model.parameters(), 0.5)
    optimizer_confid.step()
    return loss_confid.item()


# In[21]:


def dev_validate(confid_model, optimizer_confid, criterion_confid, batch, asr_model, device, DEV_BS, vocab):
    confid_model.eval()
    
    input_signal_list, input_signal_length_list, references = batch
    processed_signal_list, processed_signal_length_list = asr_model.preprocessor(
                    input_signal=input_signal_list.to(DEVICE), length=torch.tensor(input_signal_length_list, dtype=torch.float32).to(DEVICE),
                ) 
    encoded, decoder_out, soft, HYP, uncollapsed_HYP, with_blank_ids_list = ConformerForward(asr_model, processed_signal_list, processed_signal_length_list, vocab) 
    for i in range(len(HYP)):
        if HYP[i] == '':
            return 0, references, HYP, [0], [0], [0]
    
    word_ops_list, ref_j_list, hyp_j_list = fetch_word_ops(references, HYP, DEV_BS)
    word_offset_list = find_pred_ranges(with_blank_ids_list, HYP, vocab, uncollapsed_HYP)
    word_offset_list_temp = copy.deepcopy(word_offset_list) 
    actual_score_list = actual_score_gen(references, HYP, uncollapsed_HYP, word_ops_list, ref_j_list, hyp_j_list, word_offset_list_temp, soft)
    soft = soft.permute(0, 2, 1)

    
    confid_tensor = []
    final_loss = 0
    input_stack = []
    
    for i in range(len(word_offset_list)):
        for j in range(len(word_offset_list[i])):
            start_frame = word_offset_list[i][j][1]
            
            end_frame = word_offset_list[i][j][2]
            
            indices = [*range(start_frame, end_frame+1, 1)]
            # print(indices)
            encoded_input = encoded[i,:,indices]
            decoder_input = decoder_out[i,:,indices]
            soft_input = soft[i,:,indices]
            
            encoded_input = torch.mean(encoded_input, 1, keepdim=False) 
            decoder_input = torch.mean(decoder_input, 1, keepdim=False)
            soft_input = torch.mean(soft_input, 1, keepdim=False)
            
            confid_input = [encoded_input, decoder_input, soft_input]
            confid_tensor = torch.cat(confid_input, dim=-1)
            input_stack.append(confid_tensor)
    
    input_stack = torch.stack(input_stack)
    score = confid_model(input_stack)
    actual_score = [] 
    for sublist in actual_score_list:
        for item in sublist:
            actual_score.append(item)
    target_score = Variable(torch.FloatTensor(actual_score), requires_grad = True).to(device)

    if loss == 'mae':
        try:
            loss_confid = criterion_confid(score.squeeze(dim=-1), target_score)
        except Exception as e:
            print(e)
            print("REF:", references) 
            print("HYP:", HYP)
            print("OPS:", word_ops_list)
            print("TSc:", actual_score)
            print("PSc:", score)
            print(word_offset_list)
            
    if loss == 'shrinkage':
        loss_confid = criterion_confid(score.squeeze(dim=-1), target_score, a, c)
    return loss_confid.item(), references, HYP, word_ops_list, actual_score_list, score.flatten().tolist()


# In[22]:


best_loss = float('inf')

for i in tqdm(range(EPOCHS)):
    exception_counter = 0
    print(f"Epoch : {i+1}/{EPOCHS}")
    train_loss = 0
    num_train_batches = 0
    
    for batch in tqdm(trucles_trainloader):   
        train_batch_loss = train(confid_model, optimizer_confid, criterion_confid,
                           batch, asr_model, DEVICE, TRAIN_BS, vocab)
        train_loss = train_batch_loss + train_batch_loss
        num_train_batches += 1
        
    final_train_loss = train_loss/num_train_batches
   

    val_loss = 0
    num_dev_batches = 0

    for batch in tqdm(trucles_devloader):
        dev_batch_loss, references, HYP, word_ops_list, actual_score, score = dev_validate(confid_model, optimizer_confid, criterion_confid,
                                                                                           batch, asr_model, DEVICE, DEV_BS, vocab)
        val_loss += dev_batch_loss
        num_dev_batches += 1

    final_val_loss = val_loss/num_dev_batches

    # Finding the number of elements in each sublist
    num_ele_in_actual_score = [len(sublist) for sublist in actual_score]

    score_sublists = []

    start_index = 0
    for size in num_ele_in_actual_score:
        end_index = start_index + size
        sublist = score[start_index:end_index]
        score_sublists.append(sublist)
        start_index = end_index
    
    for z in range(DEV_BS):
        print("REF:", references[z]) 
        print("HYP:", HYP[z])
        print("OPS:", word_ops_list[z])
        print("TSc:", actual_score[z])
        print("PSc:", score_sublists[z])

    if final_val_loss <= best_loss:
        print('Validation loss improved, saving checkpoint.')
        best_loss = final_val_loss
        save_checkpoint_confid(confid_model, optimizer_confid, final_val_loss, i+1, 
                               best_loss_checkpoint)
    save_checkpoint_confid(confid_model, optimizer_confid, final_val_loss, i+1, 
                               checkpoint)
    print(f"Train loss : {final_train_loss}")
    print(f"Val loss : {final_val_loss}")






