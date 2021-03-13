import argparse
import datetime
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="PseudoISP")

# dataset path
# path to dataset
parser.add_argument("--datacroproot", type=str, default="../dataset/DND_mat_image/", help="path to train set")
# path to Pseudo clean dataset
# "Gaussian_Blurring_DND/" "BM3D_DND/" "DIP_DND/" "CBDNet_DND/" "RIDNet_DND/" "PT-MWRN_DND/"
parser.add_argument("--denoised_dir", type=str, default="CBDNet_DND/", help="path to denoised set")

# training setting
# Since our Pseudo-ISP is a lightweight model, patch size should not be too large
parser.add_argument("--patch_size", type=int, default=60, help="the patch size of input")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--load_thread", type=int, default=0, help="thread for data loader")

# save setting
parser.add_argument("--last_ckpt",type=str,default="none",help="the ckpt of last net")
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
parser.add_argument("--log_dir", type=str, default='./logs_DND_PseudoISP/', help='path of log files')
parser.add_argument("--save_model_freq", type=int, default=1000, help="Number of training epchs to save state")

# learning rate setting
parser.add_argument("--decay_rate", type=float, default=0.1, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=14000, help="number of epochs the model needs to run")
parser.add_argument("--learning_rate_RGB2PACK", type=float, default=1e-4, help="the initial learning rate")
parser.add_argument("--learning_rate_PACK2RGB", type=float, default=1e-4, help="the initial learning rate")
parser.add_argument("--learning_rate_Noise_Model_Network", type=float, default=1e-4, help="the initial learning rate")

opt = parser.parse_args()


