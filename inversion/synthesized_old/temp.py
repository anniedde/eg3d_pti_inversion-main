import os, shutil

models = ['experience_replay_greedy',
          'iCARL', 
          'kmeans']

percentage = '0_5'
for time in ['0', '1', '2', '3', '4', '5', 'all']:
    folder_to_copy = os.path.join(f'Margot_experience_replay_5', time, 'replay_no_lora')
    new_folder_name = f'Margot_up_to_5_ER_{percentage}'
    new_folder = os.path.join(new_folder_name, time, 'handpicked')
    # if folder_to_copy exists, copy it to new_folder
    if os.path.exists(folder_to_copy) and not os.path.exists(new_folder):
        shutil.copytree(folder_to_copy, new_folder)


for percentage in ['0_1', '0_25', '0_5']:
    new_folder_name = f'Margot_up_to_5_ER_{percentage}'
    if not os.path.exists(new_folder_name):
        os.mkdir(new_folder_name)
    for model in models:
        for time in ['0', '1', '2', '3', '4', '5', 'all']:
            folder_to_copy = os.path.join(f'Margot_up_to_5_{model}_{percentage}', time, 'replay')
            new_folder = os.path.join(new_folder_name, time, model)
            # if folder_to_copy exists, copy it to new_folder
            if os.path.exists(folder_to_copy) and not os.path.exists(new_folder):
                shutil.copytree(folder_to_copy, new_folder)

    for model in ['upper_bound', 'lower_bound']:
        for time in ['0', '1', '2', '3', '4', '5', 'all']:
            folder_to_copy = os.path.join(f'Margot_experience_replay_5', time, model)
            new_folder_name = f'Margot_up_to_5_ER_{percentage}'
            new_folder = os.path.join(new_folder_name, time, model)
            # if folder_to_copy exists, copy it to new_folder
            if os.path.exists(folder_to_copy) and not os.path.exists(new_folder):
                shutil.copytree(folder_to_copy, new_folder)

for percentage in ['0_25']:
    new_folder_name = f'Margot_up_to_4_ER_{percentage}'
    if not os.path.exists(new_folder_name):
        os.mkdir(new_folder_name)
    for model in models:
        folder_to_copy = os.path.join(f'Margot_up_to_5_{model}_{percentage}', 'replay_up_to_4')
        new_folder = os.path.join(new_folder_name, model)
        # if folder_to_copy exists, copy it to new_folder
        if os.path.exists(folder_to_copy) and not os.path.exists(new_folder):
            shutil.copytree(folder_to_copy, new_folder)
