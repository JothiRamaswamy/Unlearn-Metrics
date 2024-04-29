import json

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