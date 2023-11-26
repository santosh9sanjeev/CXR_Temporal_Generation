import torch
import matplotlib.pyplot as plt
import math
import os
class_token = False
if class_token==True:
    path = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attention_map_Test/'
    x = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attention_map_Test/cuda:0_test_output_epoch=140_2_of_2_test_main_file_v2.pt')

else:
    path = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attn_map_test_v2/'
    x = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attn_map_test_v2/cuda:0_test_output_epoch=140_2_of_2_test_main_file_v2.pt')
print(len(x))
bs = 9
sample = 4
print('len(x)', len(x))
print(x[bs]['img_paths'][sample])
attn_probs = x[bs]['attn_probs'] # get 1 batch
actual_path = x[bs]['img_paths'][sample].split('|')[1]
var = actual_path.split('/')
desired_path = '/' + '/'.join(item for item in var[1:-4]) + '/'
pt = os.path.join(desired_path,'2.0.0','files')
actual_path = pt + '/' + '/'.join(item for item in var[-4:])
print(actual_path)
attn_probs = attn_probs[sample, :, :, :-1] # get 1 sample and remove the attn_prob corresponding to cls_token key

temp_attns = attn_probs[:, 1, :1] # temporal injection
text_attns = attn_probs[:, 1, 1:257] # text
img1_attns = attn_probs[:, 1, 257:1283][:,1:-1] # img1
img2_attns = attn_probs[:, 1, 1283:2309][:,1:-1] # img2
# temp_attns = attn_probs[:, :, :1].mean(dim=1) # temporal injection
# text_attns = attn_probs[:, :, 1:257].mean(dim=1) # text
# img1_attns = (attn_probs[:, :, 257:1283].mean(dim=1))[:,1:-1] # img1
# img2_attns = (attn_probs[:, :, 1283:2309].mean(dim=1))[:,1:-1] # img2
print(img1_attns.shape)

import torchvision

nh = img1_attns.shape[0]
w = h = int(math.sqrt(img1_attns.shape[-1]))

img1_attns = torch.reshape(img1_attns, (nh, w, h))
img2_attns = torch.reshape(img2_attns, (nh, w, h))

# print(temp_attns.shape)
# print(text_attns.shape)
# print(img1_attns.shape) # (num heads, w, h)
# print(img2_attns.shape)
# print('text attnsss',text_attns)
import torch.nn.functional as F

# print(img1_attns.shape)
# img = img1_attns[10]
# print(img.shape)
# print(img.max(-1))
# tensor_cpu = img.cpu().detach().numpy()  # Move tensor to CPU and convert to NumPy array


# # Resizing the tensor to 512x512 using interpolation
# original_tensor = img  # Example random tensor of size 32x32
# resized_tensor = F.interpolate(original_tensor.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
# resized_tensor = resized_tensor.squeeze()



# # Create a matplotlib figure and plot the tensor as an image with vmax set to the tensor's max value
# plt.imshow(resized_tensor.cpu(), vmax=resized_tensor.max())  # Use cmap='gray' for grayscale images
# plt.colorbar()  # Optionally add a color bar showing the scale
# plt.savefig('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attn_map_test_v2/output_image.png')  # Save the plotted im
import torch
import matplotlib.pyplot as plt
import math
import numpy as np

# Load your attention maps or use your code to generate img1_attns and other variables
# Assuming you have img1_attns, img2_attns, and other variables already defined as in your previous code

# nh = img1_attns.shape[0]
# w = h = int(math.sqrt(img1_attns.shape[-1]))

# Plot individual attention heads in one figure
import copy
resized_tensor = torch.empty(12, 32, 32)
plt.figure(figsize=(10, 10))
for i in range(nh):
    plt.subplot(math.ceil(math.sqrt(nh)), math.ceil(math.sqrt(nh)), i+1)
    resized_tensor[i] = img1_attns[i]#F.interpolate(img1_attns[i].unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)

    plt.imshow(resized_tensor[i].cpu(), vmax=resized_tensor[i].max())
    plt.title(f"Attention Head {i+1}")
    plt.colorbar()

plt.tight_layout()
plt.savefig(path + 'output_image_all_heads.png')
plt.show()

# Combine attention maps of all heads into one image by averaging or summing them
combined_img1_attns = torch.sum(resized_tensor, dim=0)  # You can also use torch.mean() for averaging
combined_img1_attns /= nh  # Normalize by number of heads

# Convert the combined attention map tensor to a NumPy array for plotting
combined_img1_attns_numpy = combined_img1_attns.cpu().detach().numpy()

# Plot and save the combined attention maps
plt.figure(figsize=(8, 8))
plt.imshow(combined_img1_attns_numpy, vmax=combined_img1_attns_numpy.max())
plt.colorbar()  # Optionally add a color bar showing the scale
plt.savefig(path + 'output_image_combined_heads.png')
plt.show()

from PIL import Image
# actual_image = plt.imread(actual_path)
image1 = Image.open(actual_path)
image1_resized = image1.resize((32, 32))  # Replace with your desired dimensions

plt.figure(figsize=(8, 8))
plt.imshow(image1_resized,cmap='gray')
plt.savefig(path + 'actual_image.png')  # Save the combined image
plt.axis('off')  # Turn off axis labels
plt.title('Actual Image')

# Overlay attention map on top of the actual image
plt.imshow(combined_img1_attns_numpy, cmap='jet', alpha=0.5, interpolation='bilinear')
# Choose the colormap ('jet' is used as an example), adjust alpha/transparency as needed

plt.colorbar()  # Optionally add a color bar showing the scale
plt.savefig(path + 'on_image.png')  # Save the combined image

plt.show()