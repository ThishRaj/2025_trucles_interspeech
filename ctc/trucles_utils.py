import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torch.autograd import Variable

import pandas as pd
import os
import re
import gc
from numpy.testing import assert_almost_equal
import math
import numpy as np
import jiwer


def ConformerForward(encoder, decoder, char_decoder, spectrograms, input_len_list, mask, char_list, gpu=True):
    ''' Evaluate model on test dataset. '''

    encoder.eval()
    decoder.eval()

    # Move to GPU
    if gpu:
        spectrograms = spectrograms.cuda()

        mask = mask.cuda()

    with torch.no_grad():
        outputs, attention_cem = encoder(spectrograms, mask)
        outputs, decoder_cem = decoder(outputs)
 
        soft_cem = F.softmax(outputs, dim=-1)


        inds, uncollapsed_inds = char_decoder(outputs.detach(), input_len_list)
        
        uncollapsed_predictions = []
        
        for sample1 in uncollapsed_inds:
            uncollapsed_predictions.append(int_to_text_uncollapse(sample1, len(char_list), char_list))

        collapse_predictions = []
        final_predictions = []
        for sample in inds:
            collapse_predictions.append(int_to_text_collapse(sample, len(char_list), char_list))
            final_predictions.append(int_to_text_final(sample, len(char_list), char_list))

    return attention_cem, decoder_cem, soft_cem, final_predictions, uncollapsed_predictions
    



def save_checkpoint_confid(model, optimizer, valid_loss, epoch, checkpoint_path):
    ''' Save model checkpoint '''
    torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'encoder_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
