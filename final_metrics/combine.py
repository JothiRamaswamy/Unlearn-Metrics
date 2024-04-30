import json
import torch

def average_metric_dicts():
    metric_types = ['FT10_GA5', 'FT15_GA7', 'FT20_GA10']
    seeds = ['seed1_', 'seed2_', 'seed3_']
    for metric in metric_types:
        avg_dict = {}
        metric_dicts = [torch.load(seed + metric + '/all_metrics.json') for seed in seeds]
        keys = metric_dicts[0].keys()
        for k in keys:
            total = sum(d[k] for d in metric_dicts)
            avg_dict[k] = float(total/3)
            with open('final_metrics/' + metric + '.json', 'w') as file:
                json.dump(avg_dict, file)
                

with open('FT10_GA5.json', 'r') as file:
    FT10_GA5 = json.load(file)
    
with open('FT15_GA7.json', 'r') as file:
    FT15_GA7 = json.load(file)
    
with open('FT20_GA10.json', 'r') as file:
    FT20_GA10 = json.load(file)
    
total = FT10_GA5

for i, j in FT15_GA7.items():
    total[i] = j
    
for i, j in FT20_GA10.items():
    total[i] = j
    
with open('final_metrics.json', 'w') as file:
    FT20_GA10 = json.dump(total, file)
