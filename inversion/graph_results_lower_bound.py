import matplotlib.pyplot as plt
import json


times = [0, 1, 2, 3, 4]
for metric in ['lpips', 'psnr', 'dists', 'id_error']:
    # graph metric
    y = []
   #y_lora = []
    for t in times:
        tot = 0
        #tot_lora = 0
        with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/out/reconstruction-t{}.json'.format(t), 'r') as f:
            my_dict = json.load(f)
        for key in my_dict.keys():
            tot += my_dict[key][metric]
            #tot_lora += lora_results[f't{t}'][key][metric]
        y.append(tot / len(my_dict.keys()))
        #y_lora.append(tot_lora / len(lora_results[f't{t}'].keys()))
    
    plt.plot(times, y, '-o', label='no lora')
    #plt.plot(times, y_lora, '-o', label='lora')
    plt.legend()
    plt.show()
    plt.title(f'Reconstruction Eval - {metric}')
    plt.xlabel('time')
    plt.ylabel(metric)
    plt.savefig(f'out/lower_bound_eval_plot_{metric}_with_and_without_lora.png')
    plt.clf()


