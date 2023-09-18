# coding: UTF-8
"""
    @author: Yuhan Xie
    @date:   2022.7.23
    @readme: Miscellaneous utility classes and functions.
"""

import re
import importlib
import torch
import os
import sys
import types
from typing import Any, List, Tuple, Union


def SaveModel(encoder, decoder, train_loss, val_loss, optimizer, scheduler, dir, epoch):
    params = {}
    params['encoder'] = encoder.state_dict()
    params['decoder'] = decoder.state_dict()
    torch.save({'model_state_dict': params,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(dir, '_params_{}.pt'.format(epoch)))
