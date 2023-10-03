import matplotlib.pyplot as plt
import json


with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/lower_bound_interpolation_2.json', 'r') as f:
    results = json.load(f)
#with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/interpolation_lora.json', 'r') as f:
#    lora_results = json.load(f)

times = [0, 1, 2, 3, 4]

id_error_list = []
id_error_lora_list = []
for t in times:
    id_error_list.append(results[str(t)])
    #id_error_lora_list.append(lora_results[str(t)])

plt.plot(times, id_error_list, '-o', label='no lora')
#plt.plot(times, id_error_lora_list, '-o', label='lora')
plt.legend()
plt.show()
plt.title(f'Synthesis Eval - ID error')
plt.xlabel('time')
plt.ylabel('ID error')
plt.savefig(f'out/eval_lower_bound_synthesis_plot_id_error_with_and_without_lora.png')
plt.clf()