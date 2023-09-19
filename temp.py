import gdown

my_id = '1Q5BkgiablQqTciKMTryJfuijlrt2VbE9'
gdown.download_folder(id=my_id, output='/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/pti_training/coaches')

#import torch
#print(torch.cuda.is_available())