import torch
from load_model import load_model

def kl_divergence(p, q):
    # Calculate KL divergence between two normal distributions
    mu1, std1 = p.mean(), p.std()
    mu2, std2 = q.mean(), q.std()
    divergence = torch.log(std2/std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5
    return divergence

# Load the state dictionaries
model1_data = torch.load('retrain_model.pt')
model2_data = torch.load('FT_model.pt')
model3_data = torch.load('GA_model.pt')

model1_state_dict = model1_data['state_dict']
model2_state_dict = model2_data['state_dict']
model3_state_dict = model3_data['state_dict']

# Load the state dicts into the model instances
model1 = load_model(model1_state_dict)
model2 = load_model(model2_state_dict)
model3 = load_model(model3_state_dict)

model1.load_state_dict(model1_state_dict)
model2.load_state_dict(model2_state_dict)
model3.load_state_dict(model3_state_dict)

flattened_model1_states = torch.cat([t.flatten() for t in model1.state_dict().values()])
flattened_model2_states = torch.cat([t.flatten() for t in model2.state_dict().values()])
flattened_model3_states = torch.cat([t.flatten() for t in model3.state_dict().values()])

# Calculate KL divergence for each corresponding layer
total_kl_divergence = 0
for (param1, param2) in zip(model1.parameters(), model2.parameters()):
    kl_div = kl_divergence(param1.data, param2.data)
    total_kl_divergence += kl_div
    
print('Total KL Divergence:', total_kl_divergence.item())
    
total_kl_divergence = 0
for (param1, param2) in zip(model1.parameters(), model3.parameters()):
    kl_div = kl_divergence(param1.data, param2.data)
    total_kl_divergence += kl_div

print('Total KL Divergence:', total_kl_divergence.item())

