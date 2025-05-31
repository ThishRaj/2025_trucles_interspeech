import jiwer
import re
import torch

def create_token_cem_label_pairs(ground_truth, decoded_text, tokenizer, timesteps_list, softmax_probs_list):
    # Generate alignment tags using jiwer
    alignments = jiwer.visualize_alignment(
        jiwer.process_words(ground_truth.lower(), decoded_text.lower()), 
        show_measures=False, 
        skip_correct=False
    )
    alignments = alignments.split("\n")

    # Extract reference (ground truth) and hypothesis (predicted text) from alignments
    ref = alignments[1][5:]
    hyp = alignments[2][5:]

    # Clean up extra spaces and split into words
    ref = re.sub(" +", " ", ref).split()
    hyp = re.sub(" +", " ", hyp).split()

    # Generate the C, S, I, D tags
    new_ops = []
    for i in range(len(ref)):
        if hyp[i] == ref[i]:
            new_ops.append('C')
        elif '*' in hyp[i]:
            new_ops.append("D")
        elif '*' in ref[i]:
            new_ops.append("I")
        else:
            new_ops.append("S")

    # Create the (predicted_word, tag) pairs (skip 'D' tags)
    predicted_word_tag_pairs = [
        (hyp[i].replace("*", ""), new_ops[i]) 
        for i in range(len(new_ops)) if new_ops[i] != 'D'
    ]
    word_boundaries = []
    start_idx = 0
    avg_value = torch.tensor(1/1025, dtype=torch.float32).item()  # Convert to float32
    
    for word in decoded_text.split():
        word_token_ids = tokenizer.text_to_ids(word)  # Get token IDs for this word
        word_length = len(word_token_ids)
        # print(word_token_ids)
        end_idx = start_idx + word_length - 1
        if end_idx >= len(timesteps_list):  # Avoid out-of-range errors
            end_idx = len(timesteps_list) - 1
        
        word_boundaries.append((start_idx, end_idx, timesteps_list[start_idx:end_idx+1], word_token_ids)) # (token_start_idx, token_end_idx, indiced_list, token_ids_list)
        start_idx += word_length
        
    # Assign labels to tokens based on tags
    timestep_token_label = []
    for idx, (predicted_word, op) in enumerate(predicted_word_tag_pairs):
        start_idx, end_idx, indices, word_token_ids = word_boundaries[idx]
                
        if op == "C": # op is tag here
            # Take the probability of the correct token class from the softmax output
            label = 1
            true_class_probs_list = [softmax_probs_list[i][word_token_ids[j]].item() for j, i in enumerate(range(start_idx, end_idx+1))]
            timestep_token_label.extend((timestep, token, label, prob ) for token, timestep, prob in zip(word_token_ids,indices,true_class_probs_list))
                
        elif op == "S":
            # Align tokenized ground truth and hypothesis words
            gt_word_tokens = tokenizer.text_to_tokens(ref[idx])  # Get tokens for ground truth word
            pred_word_tokens = tokenizer.text_to_tokens(hyp[idx])  # Get tokens for predicted word
        
            token_alignments = jiwer.visualize_alignment(
                jiwer.process_words(" ".join(gt_word_tokens), " ".join(pred_word_tokens)),
                show_measures=False,
                skip_correct=False
            ).split("\n")
           
        
            gt_tokens = token_alignments[1][5:].split()
            pred_tokens = token_alignments[2][5:].split()

            token_ops = []
            for k in range(len(gt_tokens)):
                if pred_tokens[k] == gt_tokens[k]:
                    token_ops.append('C')
                elif '*' in pred_tokens[k]:
                    token_ops.append("D")
                elif '*' in gt_tokens[k]:
                    token_ops.append("I")
                else:
                    token_ops.append("S")
            true_class_probs = []
            label = 0
            count = 0
            for token_op in token_ops:
                try:
                    current_token = tokenizer.text_to_ids(pred_tokens[count])[0]
                    current_true_token = tokenizer.text_to_ids(pred_tokens[count])[0]
                    
                    if token_op == "C":
                        true_class_prob = softmax_probs_list[start_idx + count][current_token]
                        timestep_token_label.append((indices[count], current_token, label, true_class_prob))
                        count +=1
                    elif token_op == "S": 
                        true_class_prob = softmax_probs_list[start_idx + count][current_true_token]
                        timestep_token_label.append((indices[count], current_token, label,true_class_prob))
                        count +=1
    
                    elif token_op == "I":
                        true_class_prob = avg_value
                        timestep_token_label.append((indices[count], current_token, label, true_class_prob ))
                        count +=1
                    else:
                        pass
                except IndexError:
                    return False
                        
        elif op == "I":
            label = 0
            true_class_probs = [avg_value]*len(word_token_ids)
            timestep_token_label.extend((timestep, token, label, prob) for token, timestep, prob in zip(word_token_ids,indices,true_class_probs))

    return timestep_token_label # returns tuples of (timestep, token, label, true_class_prob)


