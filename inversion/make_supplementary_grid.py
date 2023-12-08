import os
celebs = ['Barack', 'Scarlett', 'Dwayne', 'Kamala']
test_image_names = ['140', 'test_9', 'test_5', 'test_1']
directions = ['39_Young', '39_Young', '31_Smiling', '31_Smiling']

celebs = ['Barack', 'Joe', 'Oprah', 'Scarlett', 'Xi']
test_image_names = ['54', 'test_0', 'test_1', 'test_0', 'test_1']
directions = ['31_Smiling', '31_Smiling', '31_Smiling', '31_Smiling', '31_Smiling']


celeb = 'Dwayne'
direction = '31_Smiling' #'39_Young', '31_Smiling'

test_image_name = 'test_1'

for i in range(5):
    celeb, direction, test_image_name = celebs[i], directions[i], test_image_names[i]

    for fileName in os.listdir(f'/playpen-nas-ssd/awang/semantic_editing/supplementary/{celeb}'):
        if not fileName.startswith('original'):
            model = fileName.split('-')[0]
            index = (int)(fileName.split('_')[-3])

            print('index is : ', index)
            print('model is: ', model)
            alpha_loc = f'../out/{celeb}_out/{model}/alpha/{test_image_name}.pt'
            if model == 'pretrained':
                edit_direction_path = f'../edit_directions/pretrained/{direction}.pt'
                generator_path = '/playpen-nas-ssd/awang/semantic_editing/networks/ffhqrebalanced512-128.pkl'
            else:
                edit_direction_path = f'../edit_directions/{celeb}/{model}/{direction}.pt'
                generator_path = f'/playpen-nas-ssd/awang/semantic_editing/networks/{celeb}/{model}.pkl'

            cmd = f'python ../mystyle/supplementary_grid.py --device=0 --editing_direction_path={edit_direction_path} \
                --output_dir=/playpen-nas-ssd/luchao/projects/eg3d/supplementary/results/semantic_editing/{celeb}/{model} \
                --anchor_dir=../anchors/{celeb}/ \
                    --generator_path={generator_path} \
                    --alpha_loc={alpha_loc} \
                    --edit_mag=10 \
                    --num_edits=15 \
                    --celeb={celeb} \
                    --index={index} \
                    --cameras_json_path=../data/test/{celeb}_test/cameras.json'
            
            cmd = f'python ../../my3dgen/eg3d/gen_samples.py --outdir=/playpen-nas-ssd/luchao/projects/eg3d/supplementary/results/semantic_editing/{celeb}/{model}/ \
                --trunc=0.7 --seeds=0 --reload_modules=False --shapes=True \
                --w_loc=/playpen-nas-ssd/luchao/projects/eg3d/supplementary/results/semantic_editing/{celeb}/{model}/w.pt \
                --network={generator_path} --img_name={test_image_name}'
            #os.system(cmd)
            
            cmd = f'python render_geometry.py --fname=/playpen-nas-ssd/luchao/projects/eg3d/supplementary/results/semantic_editing/{celeb}/{model}/shape.npy --sigma-threshold=10 \
                --outdir=/playpen-nas-ssd/luchao/projects/eg3d/supplementary/results/semantic_editing/{celeb}/{model}/ \
                --id_name={test_image_name} --model_name={model} \
                --original_image=/playpen-nas-ssd/awang/semantic_editing/supplementary/{celeb}/original.png'
            os.system(cmd)
    
            