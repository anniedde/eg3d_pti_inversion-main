import matplotlib.pyplot as plt
import json


with open('out/interpolation_output.json', 'r') as f:
    results = json.load(f)

for model_name in results:
    sim_list = results[model_name]
    plt.plot(range(5), sim_list, '-o', label=model_name)
plt.legend()
plt.show()
plt.title(f'Synthesis Eval - ID Similarity')
plt.xlabel('time')
plt.ylabel('ID similarity')
plt.savefig(f'out/eval_synthesis_plot.png')
plt.clf()