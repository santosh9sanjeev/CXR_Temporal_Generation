# import torch
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,8,15"
# print('hiiii')
# # Assuming you have five model paths saved
# model_paths = [
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-7-v2/epoch=97-train_loss= 4.46.ckpt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-7-temporal_token/epoch=108-train_loss= 4.44.ckpt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporal_token_125/epoch=125-train_loss= 4.41.ckpt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporal_token_129/epoch=129-train_loss= 4.41.ckpt',
#     '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporaltoken_v2/epoch=140-train_loss= 4.39.ckpt'
# ]

# # Initialize an empty state dictionary to store the averaged weights
# print('yeesssss')
# summed_weights = None

# # Load the weights from each model path and sum them up
# for path in model_paths:
#     print(path)
#     model_state = torch.load(path)  # Load model state
#     if summed_weights is None:
#         summed_weights = {key: value.cpu().numpy() for key, value in model_state.items()}
#     else:
#         # Accumulate the weights
#         for key in summed_weights.keys():
#             summed_weights[key] += model_state[key].cpu().numpy()

# # Calculate the average by dividing by the number of models
# num_models = len(model_paths)
# averaged_weights = {key: value / num_models for key, value in summed_weights.items()}

# # Convert the averaged weights back to torch tensors
# averaged_weights = {key: torch.from_numpy(value) for key, value in averaged_weights.items()}


# # Now 'averaged_weights' contains the averaged weights from the five models
# # You can save these averaged weights to use them for further processing or inference
# torch.save(averaged_weights, '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/model_soups/averaged_weights.pth')
# print('asdalskfj')


import torch
import copy
# Assuming you have five models initialized or loaded
model_1 = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-7-v2/epoch=97-train_loss= 4.46.ckpt')
model_2 = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-7-temporal_token/epoch=108-train_loss= 4.44.ckpt')
model_3 = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporal_token_125/epoch=125-train_loss= 4.41.ckpt')
model_4 = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporal_token_129/epoch=129-train_loss= 4.41.ckpt')
model_5 = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/temporaltoken_v2/epoch=140-train_loss= 4.39.ckpt')
print('hiiiiiiiiiiiiiiiiii')
# Get the state dictionaries of the models
state_dict_1 = model_1['state_dict']
state_dict_2 = model_2['state_dict']
state_dict_3 = model_3['state_dict']
state_dict_4 = model_4['state_dict']
state_dict_5 = model_5['state_dict']
print('dasjdklasjfdsvkl')

# Initialize an empty state dictionary to store the averaged weights
# averaged_state_dict = {}

# # Iterate through the keys in the state dictionaries and average the weights
# for key in state_dict_1.keys():
#     # Assuming all models have the same keys
#     averaged_state_dict[key] = (state_dict_1[key] + state_dict_2[key] + state_dict_3[key] +
#                                 state_dict_4[key] + state_dict_5[key]) / 5.0

# from unified_plmodel import TransformerLightning_unified

# model = TransformerLightning_unified(
#     lr=args.lr,
#     weight_decay=args.weight_decay,
#     tokenizer=tokenizer,
#     pad_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"),
#     sos_token_idx=tokenizer.convert_tokens_to_ids("[CLS]"),
#     eos_token_idx=tokenizer.convert_tokens_to_ids("[SEP]"),
#     # save_dir='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-5-CXRBERT_cls_token',
#     save_dir=args.save_dir,
#     causal_trans=args.causal_clm,
#     **kargs_unified,
# )


# # Create a new model with the averaged weights
# averaged_model = #'/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/model_soups/averaged_weights.pth'  # Create a new instance of the model with the same architecture
# averaged_model.load_state_dict(averaged_state_dict)

# averaged_state_dict = {}
averaged_state_dict = copy.deepcopy(model_1)

# Iterate through the keys in the state dictionaries and average the weights
for key in state_dict_1.keys():
    averaged_state_dict['state_dict'][key] = (state_dict_1[key] + state_dict_2[key] + state_dict_3[key] +
                                state_dict_4[key] + state_dict_5[key]) / 5.0

# Save the averaged weights to a new file
torch.save(averaged_state_dict, '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/model_soups/averaged_weights_v2.pt')

"""
This code is under adaptation by SANZ
"""


# import os
# import argparse
# import datetime
# from functools import partial
# import time
# import torch.nn as nn
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# from tokenizers import ByteLevelBPETokenizer
# from tokenizers.processors import BertProcessing  

# from helpers import str2bool
# from datamodule import CXRDataModule
# from loader_unified import UnifiedCXRDataset
# from unified_plmodel import TransformerLightning_unified
# import warnings
# from transformers import AutoModel, AutoTokenizer

# warnings.filterwarnings("ignore")

# # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,10,11,12,13"

# # os.environ["CUDA_VISIBLE_DEVICES"] = "12"



# cache_direc = "./biomed_VLP/"
# # Load the model and tokenizer
# url = "microsoft/BiomedVLP-CXR-BERT-specialized"

# if __name__ == '__main__':
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--seed', default=42, type=int)

#     parser.add_argument('--test', default=False, help='trian (False) or test (True)')
#     parser.add_argument('--reload_ckpt_dir', default=None, type=str, help='ckpt_dir')
#     parser.add_argument('--save_dir', default=None, type=str, help='save_dir')

#     parser.add_argument('--n_gpus', default=1, type=int)
#     parser.add_argument('--n_epochs', default=200, type=int)
#     parser.add_argument('--batch_size', default=6, type=int)
#     parser.add_argument('--lr', default=4.5e-6, type=float, help='learning rate')
#     parser.add_argument('--accumulate_grad_batches', default=1, type=float)
#     parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')

#     parser.add_argument('--img_root_dir', default='/nfs/users/ext_ibrahim.almakky/datasets/physionet.org/files/mimic-cxr-jpg/', type=str)
#     parser.add_argument('--text_root_dir', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/mimic-cxr-reports-v4', type=str)#/nfs/users/ext_ibrahim.almakky/datasets/physionet.org/files/mimic-cxr-jpg/mimic-cxr-reports-v2/
#     parser.add_argument('--train_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/train_main_file_v2.csv', type=str)
#     parser.add_argument('--val_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/validate_main_file_v2.csv', type=str)
#     parser.add_argument('--test_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/test_main_file_v2.csv', type=str)

#     # parser.add_argument('--text_root_dir', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/mimic-cxr-reports-v3', type=str)#/nfs/users/ext_ibrahim.almakky/datasets/physionet.org/files/mimic-cxr-jpg/mimic-cxr-reports-v2/
#     # parser.add_argument('--train_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v2/train_main_file.csv', type=str)
#     # parser.add_argument('--val_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v2/validate_main_file.csv', type=str)
#     # parser.add_argument('--test_meta_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v2/test_main_file_smaller_version.csv', type=str)

#     parser.add_argument('--vocab_file', default='BBPE_tokenizer/vocab.json', type=str)
#     parser.add_argument('--merge_file', default='BBPE_tokenizer/merges.txt', type=str)

#     parser.add_argument('--vqgan', default=512, type=int, help='vqgan img resolution')
#     parser.add_argument('--vqgan_model_path', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/mimic_vqgan/last.ckpt', type=str)
#     parser.add_argument('--vqgan_config_path', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/mimic_vqgan/2021-12-17T08-58-54-project.yaml', type=str)
#     parser.add_argument('--codebook_indices_path', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/mimic_vqgan/mimiccxr_vqgan1024_res512_codebook_indices.pickle', type=str)


#     parser.add_argument('--max_img_num', default=2, type=int, help='must be less than or equal to target_count')
#     parser.add_argument('--target_count', default=2, type=int, help='select target goup, S w/1, w/2, w/3')
#     parser.add_argument('--under_sample', default='fixed_all_unified', type=str)
#     parser.add_argument('--max_text_len', default=256, type=int)
#     parser.add_argument('--target_view', default=['AP', 'PA', 'LATERAL', 'LL'], nargs='+', type=str)

#     parser.add_argument('--transformer', default=True)
#     parser.add_argument('--FAVOR', default=True)
#     parser.add_argument('--generalized_attention', default=True, help='defaults to softmax approximation, but can be set to True for generalized attention')
#     parser.add_argument('--dim', default=768, type=int, help='dimension. dimension must be divisible by number of heads.')
#     parser.add_argument('--depth', default=12, type=int, help='layers')
#     parser.add_argument('--heads', default=12, type=int, help='heads')
#     parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
#     parser.add_argument('--dim_head', default=64, type=int, help='dim of head. inner_dim = dim_head * heads')
#     parser.add_argument('--local_attn_heads', default=0, type=int, help='if n heads are local attention, heads-n others are global performers.')
#     parser.add_argument('--local_window_size', default=256, type=int, help='window size of local attention')
#     parser.add_argument('--causal', default=True, type=str2bool, help='auto-regressive or not')
#     parser.add_argument('--attn_type', default='all_modality_causal_cuda', type=str)
#     parser.add_argument('--causal_clm', default='conditioned_causal', choices=['conditioned_causal', 'causal'], type=str, help='Not in used')
#     parser.add_argument('--nb_features', default=64, type=int, help='number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head.')
#     parser.add_argument('--feature_redraw_interval', default=1000, type=int,
#                         help='how frequently to redraw the projection matrix, the more frequent, the slower the training')
#     parser.add_argument('--reversible', default=False, type=str2bool,
#                         help='reversible layers, from Reformer paper. Works only when sharded_ddp=True')
#     parser.add_argument('--ff_chunks', default=1, type=int, help='chunk feedforward layer, from Reformer paper')
#     parser.add_argument('--ff_glu', default=False, type=str2bool, help='use GLU variant for feedforward')
#     parser.add_argument('--emb_dropout', default=0.1, type=float, help='embedding dropout')
#     parser.add_argument('--ff_dropout', default=0.1, type=float, help='feedforward dropout')
#     parser.add_argument('--attn_dropout', default=0.1, type=float, help='post-attn dropout')
#     parser.add_argument('--use_scalenorm', default=False, type=str2bool,
#                         help='use scale norm, from Transformers without Tears paper')
#     parser.add_argument('--use_rezero', default=False, type=str2bool,
#                         help='use rezero, from Rezero is all you need paper')
#     parser.add_argument('--tie_embed', default=False, type=str2bool,
#                         help='multiply final embeddings with token weights for logits')
#     parser.add_argument('--rotary_position_emb', default=False, type=str2bool,
#                         help='use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding')


#     parser.add_argument('--save_top_k', default=1, type=int)
#     parser.add_argument('--num_workers', default=8, type=int)
#     parser.add_argument('--gradient_clip_val', default=0, type=float)
#     parser.add_argument('--num_sanity_val_steps', default=2, type=int)
#     parser.add_argument('--fp16', default=False, type=str2bool, help='FP16')
#     parser.add_argument('--sharded_ddp', default=False, type=str2bool, help='fairscale sharded ddp')

#     parser.add_argument('--train_label_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/train_labels_file_v2.csv', type=str)
#     parser.add_argument('--validate_label_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/validate_labels_file_v2.csv', type=str)
#     parser.add_argument('--test_label_file', default='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/data/metadata_v3/test_labels_file_v2.csv', type=str)
#     parser.add_argument('--weights', default=True, type=str2bool, help='weights be given or not')

#     args = parser.parse_args()

#     start = datetime.datetime.now()
#     TODAY = str(datetime.date.today().strftime('%Y%m%d'))
#     NOW = str(start.strftime('_%Hh%Mm%Ss'))
#     print("\n")
#     pl.seed_everything(args.seed, workers=True)

#     # tokenizer = ByteLevelBPETokenizer(
#     #     args.vocab_file,
#     #     args.merge_file,
#     # )
#     # tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
#     # tokenizer._tokenizer.post_processor = BertProcessing(
#     #     ("[EOS]", tokenizer.token_to_id("[EOS]")),
#     #     ("[SOS]", tokenizer.token_to_id("[SOS]")),
#     # )
#     # tokenizer.enable_truncation(max_length=args.max_text_len)
#     # tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=args.max_text_len)  

#     tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True, cache_dir = cache_direc)
#     tokenizer.add_special_tokens({"additional_special_tokens":["[PAD]", "[CLS]", "[SEP]", "[MASK]"]}) #sansan
#     print('lennnnn of tokeniserrr', len(tokenizer))


#     dsclass = partial(
#         UnifiedCXRDataset,
#         img_root_dir=args.img_root_dir,
#         text_root_dir=args.text_root_dir,
#         vqgan_model_path=args.vqgan_model_path,
#         vqgan_config_path=args.vqgan_config_path,
#         codebook_indices_path=args.codebook_indices_path,
#         vqgan=args.vqgan,
#         max_img_num=args.max_img_num,
#         max_text_len=args.max_text_len,
#         tokenizer=tokenizer,
#         target_count=args.target_count,
#         target_view=args.target_view,
#         under_sample=args.under_sample,
#     )

#     train_ds = dsclass(args.train_meta_file,label_file = args.train_label_file)
#     val_ds = dsclass(args.val_meta_file,label_file = args.validate_label_file)
#     test_ds = dsclass(args.test_meta_file,label_file = args.test_label_file)

#     dm = CXRDataModule(
#         train_ds, val_ds, test_ds,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers
#     )

#     args.num_tokens = train_ds.text_vocab_size
#     args.img_vocab_size = train_ds.img_vocab_size

#     args.num_img_tokens = train_ds.img_vocab_size + (2*2) + args.max_img_num

#     args.max_text_len = train_ds.max_text_len  # 256
#     args.max_img_len = train_ds.img_len * args.max_img_num
#     args.max_seq_len = train_ds.img_len * args.max_img_num + train_ds.max_text_len

#     args.img_len = train_ds.img_len
#     args.max_img_num = args.max_img_num
#     args.img_fmap_size = int(train_ds.img_fmap_size)

#     print("\n max_seq_len ", args.max_seq_len, "\n")
#     print(" num_img_tokens ", args.num_img_tokens, "\n")
#     print(" max_img_len ", args.max_img_len, "\n")
#     print(" num_tokens(text_vocab_size) ", args.num_tokens, "\n")
#     print(" max_text_len ", args.max_text_len, "\n")


#     kargs_unified = {
#         'num_tokens': args.num_tokens,
#         'num_img_tokens': args.num_img_tokens,
#         'img_vocab_size':args.img_vocab_size, 
#         'max_seq_len': args.max_seq_len,
#         'max_img_len' : args.max_img_len,
#         'max_img_num': args.max_img_num,
#         'img_len':args.img_len,
#         'dim': args.dim,
#         'depth': args.depth,
#         'heads': args.heads,
#         'dim_head': args.dim_head,
#         'local_attn_heads': args.local_attn_heads,
#         'local_window_size': args.local_window_size,
#         'causal': args.causal,
#         'attn_type': args.attn_type,
#         'nb_features': args.nb_features,
#         'feature_redraw_interval': args.feature_redraw_interval,
#         'reversible': args.reversible,
#         'ff_chunks': args.ff_chunks,
#         'ff_glu': args.ff_glu,
#         'emb_dropout': args.emb_dropout,
#         'ff_dropout': args.ff_dropout,
#         'attn_dropout': args.attn_dropout,
#         'generalized_attention': args.generalized_attention,
#         'kernel_fn': nn.ReLU(),
#         'use_scalenorm': args.use_scalenorm,
#         'use_rezero': args.use_rezero,
#         'tie_embed': args.tie_embed,
#         'rotary_position_emb': args.rotary_position_emb,
#         'img_fmap_size': args.img_fmap_size,
#         'FAVOR': args.FAVOR,
#         'epochs': args.n_epochs,
#         'ckpt_dir': args.reload_ckpt_dir,
#         'under_sample': args.under_sample,
#         'target_count': args.target_count,
#         'weights':args.weights
#     }

#     LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        

#     model = TransformerLightning_unified(
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#         tokenizer=tokenizer,
#         pad_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"),
#         sos_token_idx=tokenizer.convert_tokens_to_ids("[CLS]"),
#         eos_token_idx=tokenizer.convert_tokens_to_ids("[SEP]"),
#         # save_dir='/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/trained_models/exp-5-CXRBERT_cls_token',
#         save_dir=args.save_dir,

#         causal_trans=args.causal_clm,
#         **kargs_unified,
#     )