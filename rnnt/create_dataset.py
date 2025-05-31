#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from create_token_cem_labels import create_token_cem_label_pairs
from tqdm import tqdm
import pandas as pd
import numpy as np



def get_embeddings(k,timesteps_list,enc_outs):
    '''
    Takes the encoder outputs of a single wav file sample(in the form of log_probs),
    use the emission times present in timesteps_list and extract the 2k+1 frames' of log_probs for each emitted token
    '''
    extracted_frames = []
    
    # Iterate through each token emission time
    for timestep in timesteps_list:
        start_idx = max(0, timestep - k)  # Ensure we don't go below 0
        end_idx = min(enc_outs.size(0), timestep + k + 1)  # Ensure we don't go out of bounds
        frames = enc_outs[start_idx:end_idx]  # Extract the frames
        extracted_frames.append(frames)
    return extracted_frames



def post_processing_decoder_predictions(timesteps_list, decoder_debug_tensors):
    aligned_decoder_debug = []
    debug_idx = 0  # Index to track decoder_debug_tensors

    if not timesteps_list:
        return aligned_decoder_debug

    for i in range(len(timesteps_list)):
        try:
            if i > 0 and timesteps_list[i] == timesteps_list[i - 1]:
                if not aligned_decoder_debug:
                    raise ValueError(" Error: First timestep is a duplicate but no previous embedding exists!")
                aligned_decoder_debug.append(aligned_decoder_debug[-1].squeeze())  
            else:
                aligned_decoder_debug.append(decoder_debug_tensors[i].squeeze())

        except as e:
            print(f"Exception at i={i}: {e}")
            print(f"debug_idx={debug_idx}, len(decoder_debug_tensors)={len(decoder_debug_tensors)}")
            print(f"timesteps_list length={len(timesteps_list)}, timesteps_list={timesteps_list}")
            # return None
            break  # Stop further processing to prevent corruption

    return aligned_decoder_debug



def extract_acoustic_features(list_enc_embeddings, decoder_debug_tensors):
    '''
    (1) Takes the list of encoder embeddings(each element of size torch.Size([3, 640])) and list of decoder embedding(each element of size torch.Size([640]))
    (2) Computes the element wise(list's element) cross-attention with 
    decoder tensor as query and 2k+1(k=1 here) encoder embeddings(present in each element) as key,
    (3) Multiplies the softmax of attention scores with corresponding 2k+1 embeddings of element of encoder embeddings list, and then
    (4) Appends decoder tensor to the result getting a flattened tensor of shape torch.Size([2560]) 4*640=2560, i.e., 3*640 + 640
    '''
    total_acoustic_features = []
    with torch.inference_mode():
        for i in range(len(decoder_debug_tensors)):
            attention_scores = torch.matmul(list_enc_embeddings[i], decoder_debug_tensors[i].unsqueeze(1))  # Shape: [3, 1]
            # Apply softmax to the attention scores to get attention weights
            attention_weights = F.softmax(attention_scores, dim=0)  # Shape: [3, 1]
            
            # Multiply the attention weights with their corresponding sub-sequence vectors
            weighted_sub_sequence = list_enc_embeddings[i] * attention_weights  # Shape: [3, 640] (broadcasting works automatically)
            
            # Flatten the result into a single vector of shape [3 * 640]
            acoustic_feature = weighted_sub_sequence.flatten()  # Shape: [1920]
            acoustic_feature = torch.cat([acoustic_feature, decoder_debug_tensors[i]], dim=0)  # Shape: [2560]
            # it will return acoustic_feature of shape torch.Size([2560])
            total_acoustic_features.append(acoustic_feature)
    return total_acoustic_features
        
        



with open("./data_gen_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader) 
device = 'cuda'
model_path = config['asr_model_path']
metadata_path = config["metadata_path"]
root_dir = config["audio_root_dir"]
pickle_file_path = config["pickle_file_save_path"]


# Load the RNNT model
asr_model = EncDecRNNTBPEModel.restore_from(model_path)

# Set the device (GPU if available, else CPU)
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)
asr_model.eval()  # Ensures correct behavior of BatchNorm/Dropout

import pandas as pd
# meta_path = "./data/ASR_datasets/CVE/metdata.tsv" # change for librispeech
metadata = pd.read_csv(metadata_path, sep="\t", header=None) # audio_path, text, duration(secs)
# print(metadata.head(2))

metadata[0] = root_dir + "/" + metadata[0].astype(str)


# Remove rows with NaN or invalid duration values
metadata = metadata.dropna(subset=[2])

# Convert the duration column (Index 2) to numeric values
metadata[2] = pd.to_numeric(metadata[2], errors='coerce')

# Check for NaN after conversion
print("NaN values in duration column:", metadata[2].isnull().sum())


# Filter rows where duration (Column 2) is between 2.0 and 15.0 seconds
metadata = metadata[(metadata[2] > 2.0) & (metadata[2] < 13.5)]

all_data = []
# Open the pickle file in append mode
with open(pickle_file_path, 'ab') as f:
    # Loop over each row in the metadata file
    for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing Audio Files"):
        audio_file = row[0]  # Path to the audio file
        ground_truth = row[1]  # Corresponding ground truth text
    
        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_file)
        audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        audio_tensor = audio_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
        audio_tensor = audio_tensor.squeeze(0)  # Remove extra dimension, shape: (1, time)
    
        # Forward pass through the model
        with torch.inference_mode():
            enc_outs, encoded_lengths = asr_model.forward(
                input_signal=audio_tensor,
                input_signal_length=torch.tensor([audio_tensor.shape[1]]).to(device)
            )
    
            # Decode the predictions using RNNT-specific decoding method
            decoded_text, timesteps_list, decoder_debug_tensors, _ = asr_model.decoding.rnnt_decoder_predictions_tensor(enc_outs, encoded_lengths) # modified the source code into returning these
            
            timesteps_list = timesteps_list.tolist()
            decoder_debug_tensors = [t.squeeze().reshape(640) for t in decoder_debug_tensors[:-1]]
            len_timestep_list = len(timesteps_list)
            len_decoder_debug_tensors = len(decoder_debug_tensors)
            if len_timestep_list != len_decoder_debug_tensors or  len_decoder_debug_tensors == 0 or len_timestep_list == 0:
                continue
            enc_outs = enc_outs.permute(0, 2, 1)  # Reshape (B, T, D) -> (1, 168, 256)
            filtered_enc_outs = [enc_outs.squeeze(0)[i] for i in timesteps_list]
            enc_outs_list = [asr_model.joint.enc(i) for i in filtered_enc_outs]
            enc_dec_list = [enc + dec for enc, dec in zip(enc_outs_list, decoder_debug_tensors)]
            probs_list = [asr_model.joint.joint_net(i) for i in enc_dec_list] # predicted probabilities tensor for each token emission.
            softmax_list = [torch.nn.functional.softmax(tens, dim=0) for tens in probs_list]

            enc_outs = asr_model.joint.enc(enc_outs)  # Pass through the 'enc' layer
            enc_outs = enc_outs.permute(0, 2, 1)  # Reshape back to (B, D', T), i.e., (1, 640, 168)
    
            # Create token-level pairs
            list_emisn_t_token_label = create_token_cem_label_pairs(ground_truth=ground_truth, decoded_text=decoded_text[0].lower(), timesteps_list=timesteps_list, tokenizer=asr_model.tokenizer, softmax_probs_list = softmax_list)
            if not list_emisn_t_token_label:
                continue
            
            # Obtain encoder embeddings
            encoder_embeddings_list = get_embeddings(k=1, timesteps_list=timesteps_list, enc_outs=enc_outs.squeeze(0).transpose(0, 1))
            
            # Extract total acoustic features
            total_acoustic_features = extract_acoustic_features(encoder_embeddings_list, decoder_debug_tensors)

            pickle.dump({'acoustic_features': total_acoustic_features, 'emission_time_label': list_emisn_t_token_label }, f, protocol=pickle.HIGHEST_PROTOCOL) # storing token wise features for each sentence 
