""" DEFAULT PROJECT-WIDE SETTINGS """

import torch 
import numpy as np
import os
import re

torch.set_default_device('cpu')          # compute on CPU
torch.set_default_dtype(torch.float64)   # double precision for numerical stability
torch.manual_seed(42)                    # reproducibility
