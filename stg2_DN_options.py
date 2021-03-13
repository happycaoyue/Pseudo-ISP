
import argparse
import torch

parser = argparse.ArgumentParser(description="PseudoISP")
#
parser.add_argument("--valid_flag", type=int, default=0, help="1 with Pseudo paried as validation 0 without Pseudo paried as validation ")
# Orginal Noisy DataSet Setting
parser.add_argument("--dataset_path", type=str, default="../dataset/DND_mat_image/", help="path to val set")
parser.add_argument("--Noisy_path", type=str, default="noisy_mat/", help="path to val set")
parser.add_argument("--Denoised_path", type=str, default="CBDNet_DND/", help="path to val set")
# Clean DataSet Setting
parser.add_argument("--Div_path", type=str, default='../dataset/DIV2K_train_HR/',help="DIV2K path")

# Training Setting
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--Synthesis_size", type=int, default=16, help="Synthesis data of Training batch size, must <= batch_size")
parser.add_argument("--load_thread", type=int, default=0, help="thread for data loader")


# Synthesis Dataset (Synthesis Noisy Images, Clean Images)
parser.add_argument("--traindbf", type=str, default='./dataset_h5py/Synthesis_Div_p160_s60_srgb_uint8.h5', help="Synthesis h5 file")
# Previous Dataset (testing Noisy Images, Previous Denoised Images)
parser.add_argument("--traindbf_pre", type=str, default='./dataset_h5py/Pseudo_Clean_srgb_uint8.h5', help="Pseudo Clean (denoisied) h5 file")

# Save Setting
# Automatic generation requires no setup
parser.add_argument("--last_ckpt",type=str, default="/dn_CBDNet_sRGB_e47.pth",help="the ckpt of last net")
# Automatic generation requires no setup
parser.add_argument("--resume", type=str, choices=("continue", "new"), default="new",help="continue to train model")
# "dn_MWCNN_sRGB_e/" "dn_CBDNet_sRGB_e/" "dn_RIDNet_sRGB_e/" "dn_MWRN_sRGB_e/"
parser.add_argument("--save_prefix", type=str, default="dn_CBDNet_sRGB_e",help="prefix added to all ckpt to be saved")
parser.add_argument("--save_every_epochs", type=int, default=1, help="Number of training epchs to save state")
parser.add_argument("--save_path", type=str, default='./net_last_ckpt/',help="prefix added to all ckpt to be saved")

# Training setting
# Different denoising methods should have different Settings
parser.add_argument("--learning_rate_dn", type=float, default=1e-4, help="the initial learning rate")
parser.add_argument("--decay_rate", type=float, default=0.1, help="the decay rate of lr rate")
parser.add_argument("--epoch", type=int, default=300, help="number of epochs the model needs to run")
parser.add_argument("--steps", type=list, default=[20, 200], help="schedule steps,use comma(,) between numbers")
parser.add_argument("--ISP_num", type=int, default = 50, help="number of PseudoISP")
parser.add_argument("--PseudoISP_path", type=str, default='./PseudoISP_ckpt/',help="PseudoISP path")


opt = parser.parse_args()



