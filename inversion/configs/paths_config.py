## Pretrained models paths
eg3d_ffhq = 'networks/orange_cat/pretrained.pkl'
dlib = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils/align.dat'

## Dirs for output files
checkpoints_dir = './checkpoints'
embedding_base_dir = './embeddings'
experiments_output_dir = './output'
logdir = './logs'

## Input info
# Location of the cameras json file
input_pose_path = '/playpen-nas-ssd/awang/data/luchao_test_minus_subset/epoch_20_000000/cameras.json'
# The image tag to lookup in the cameras json file
input_id = 'IMG_7716'
# Where the input image resides
input_data_path = '/playpen-nas-ssd/awang/data/orange_cat_test'
# Where the outputs are saved (i.e. embeddings/{input_data_id})
embedding_folder = 'luchao_subset/no_lora'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'
