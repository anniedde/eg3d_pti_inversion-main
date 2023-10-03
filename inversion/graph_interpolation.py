import matplotlib.pyplot as plt
import json


with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/interpolation.json', 'r') as f:
    results = json.load(f)
with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/interpolation_lora.json', 'r') as f:
    lora_results = json.load(f)
with open('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/lower_bound_interpolation_2.json', 'r') as f:
    lower_bound_results = json.load(f)

times = [0, 1, 2, 3, 4]

id_error_list = []
id_error_lower_bound_list = []
for t in times:
    id_error_list.append(results[str(t)])
    id_error_lower_bound_list.append(lower_bound_results[str(t)])

plt.plot(times, id_error_list, '-o', label='upper bound')
plt.plot(times, id_error_lower_bound_list, '-o', label='lower bound')
plt.legend()
plt.show()
plt.title(f'Synthesis Eval - ID error')
plt.xlabel('time')
plt.ylabel('ID error')
plt.savefig(f'out/eval_synthesis_plot_id_error_lower_vs_upper_bound.png')
plt.clf()