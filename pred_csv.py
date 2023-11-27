import torch
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
class_token = True
if class_token==True:
    path = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/test_classifier/cuda:0_test_output_epoch=140_1_of_2_test_main_file_v2.pt'
    x = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/test_classifier/cuda:0_test_output_epoch=140_1_of_2_test_main_file_v2.pt')

else:
    path = '/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attn_map_test_v2/'
    x = torch.load('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/attn_map_test_v2/cuda:0_test_output_epoch=140_2_of_2_test_main_file_v2.pt')

bs = len(x)
print(len(x),len(x[0]))
preds = []
study_ids = []
subject_ids = []
dicom_ids = []
for i in range(bs):
    ls = list(x[i]['cls_logits'].to('cpu'))
    ls1 = [tensor.tolist() for tensor in ls]
    if i == bs-1:
        batch_size = 2
    else:
        batch_size=2
    for j in range(batch_size):
        print(j)
        cls_vals = ls1[j]
        actual_path = x[i]['img_paths'][j].split('|')[0]
        var = actual_path.split('/')
        dicom_id = var[-1][:-4]
        subject_id = x[i]['subject_ids'][j]
        study_id = var[-2]
        preds.append(cls_vals)
        dicom_ids.append(dicom_id)
        subject_ids.append(subject_id)
        study_ids.append(study_id)


# print(study_ids)
print(len(preds))
print(len(dicom_ids))
print(len(subject_ids))
print(len(study_ids))

# print(preds)
data = {
    'dicom_id': dicom_ids,
    'subject_id': subject_ids,
    'study_id': study_ids
}
columns = ['dicom_id', 'subject_id', 'study_id']

# Add columns for disease classes
disease_classes = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum',
    'Fracture', 'Lung_Lesion', 'Lung_Opacity', 'Pleural_Effusion', 'Pleural_Other',
    'Pneumonia', 'Pneumothorax', 'Support_Devices'
]

for i, disease in enumerate(disease_classes):
    data[disease] = [row[i] for row in preds]

# Create the DataFrame
df = pd.DataFrame(data, columns=columns + disease_classes)
df.to_csv('/nfs/users/ext_ibrahim.almakky/Santosh/CVPR/temporal_project/UniXGen/classifier_csvs/test_1_2.csv')
# Display the DataFrame
print(df)
