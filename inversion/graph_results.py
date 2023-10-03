import matplotlib.pyplot as plt
import json


with open('/playpen-nas-ssd/awang/data/luchao_test_10_images/reconstruction_results.json', 'r') as f:
    results = json.load(f)
with open('/playpen-nas-ssd/awang/data/luchao_test_10_images/reconstruction_with_lora_results.json', 'r') as f:
    lora_results = json.load(f)

times = [1, 2, 3, 4, 5]
for metric in ['lpips', 'psnr', 'dists', 'id_error']:
    # graph metric
    y = []
    y_lora = []
    for t in times:
        tot = 0
        tot_lora = 0
        for key in results[f't{t}'].keys():
            tot += results[f't{t}'][key][metric]
            tot_lora += lora_results[f't{t}'][key][metric]
        y.append(tot / len(results[f't{t}'].keys()))
        y_lora.append(tot_lora / len(lora_results[f't{t}'].keys()))
    
    plt.plot(times, y, '-o', label='no lora')
    plt.plot(times, y_lora, '-o', label='lora')
    plt.legend()
    plt.show()
    plt.title(f'Reconstruction Eval - {metric}')
    plt.xlabel('time')
    plt.ylabel(metric)
    plt.savefig(f'out/eval_plot_{metric}_with_and_without_lora.png')
    plt.clf()


