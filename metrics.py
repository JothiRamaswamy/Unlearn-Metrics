import torch
from load_model import *
from scipy.stats import wasserstein_distance, entropy
import numpy as np

def get_wasserstein_distance(tensor1, tensor2):
    dists = []
    for i in range(len(tensor1)):
        dists.append(wasserstein_distance(tensor1[i], tensor2[i]))
    return sum(dists)/len(tensor1)

def get_kl_divergence(hist1, hist2):
    prob_dist1 = hist1 / hist1.sum()
    prob_dist2 = hist2 / hist1.sum()
    def smooth_distribution(distribution, epsilon=1e-45):
    # Adds epsilon to the distribution and normalizes it
        return (distribution + epsilon) / (1 + epsilon * len(distribution))
    smooth_prob_dist1 = smooth_distribution(prob_dist1)
    smooth_prob_dist2 = smooth_distribution(prob_dist2)
    # Calculate the KL divergence
    return entropy(smooth_prob_dist1, smooth_prob_dist2)

def get_metrics(directory):
    final_dict = {}
    os.chdir(directory)
    def load_previous_metrics():
        model_types = ['retrain', 'FT', 'GA']
        for model_type in model_types:
            filename = model_type + '_metrics.json'
            metrics = json.loads(torch.load(filename))
            final_dict[model_type + '_retain_accuracy'] = metrics['accuracy']['retain']
            final_dict[model_type + '_forget_accuracy'] = metrics['accuracy']['forget']
            final_dict[model_type + '_val_accuracy'] = metrics['accuracy']['val']
            final_dict[model_type + '_test_accuracy'] = metrics['accuracy']['test']
            final_dict[model_type + '_train_privacy'] = metrics['SVC_MIA_training_privacy']['correctness']
            final_dict[model_type + '_forget_efficacy'] = metrics['SVC_MIA_forget_efficacy']['correctness']
    load_previous_metrics()

    # Load the state dictionaries
    orig_model_data = torch.load('model.pt')
    retrain_data = torch.load('retrain_model.pt')
    FT_data = torch.load('FT_model.pt')
    GA_data = torch.load('GA_model.pt')

    #Test predictions
    orig_model_pred = torch.load('predictions/orig_test_pred_logits.pt')
    retrain_pred = torch.load('predictions/retrain_test_pred_logits.pt')
    FT_pred = torch.load('predictions/FT_test_pred_logits.pt')
    GA_pred = torch.load('predictions/GA_test_pred_logits.pt')

    #Forget predictions
    orig_forget_pred = torch.load('predictions/orig_forget_pred_logits.pt')
    retrain_forget_pred = torch.load('predictions/retrain_forget_pred_logits.pt')
    FT_forget_pred = torch.load('predictions/FT_forget_pred_logits.pt')
    GA_forget_pred = torch.load('predictions/GA_forget_pred_logits.pt')

    orig_model_state_dict = orig_model_data['state_dict']
    retrain_state_dict = retrain_data['state_dict']
    FT_state_dict = FT_data['state_dict']
    GA_state_dict = GA_data['state_dict']

    # Load the state dicts into the model instances
    orig_model = load_model(orig_model_state_dict)
    retrain = load_model(retrain_state_dict)
    FT = load_model(FT_state_dict)
    GA = load_model(GA_state_dict)

    orig_model.load_state_dict(orig_model_state_dict)
    retrain.load_state_dict(retrain_state_dict)
    FT.load_state_dict(FT_state_dict)
    GA.load_state_dict(GA_state_dict)

    flattened_orig_model_states = torch.cat([t.flatten() for t in orig_model.state_dict().values()])
    flattened_retrain_states = torch.cat([t.flatten() for t in retrain.state_dict().values()])
    flattened_FT_states = torch.cat([t.flatten() for t in FT.state_dict().values()])
    flattened_GA_states = torch.cat([t.flatten() for t in GA.state_dict().values()])

    orig_model_hist = torch.histc(flattened_orig_model_states, bins=4000, min=-0.5, max=1.5)
    retrain_hist = torch.histc(flattened_retrain_states, bins=4000, min=-0.5, max=1.5)
    FT_hist = torch.histc(flattened_FT_states, bins=4000, min=-0.5, max=1.5)
    GA_hist = torch.histc(flattened_GA_states, bins=4000, min=-0.5, max=1.5)

    ###WEIGHT SPACE###
    #L2 Distance between weights
    retrain_FT_dist = torch.dist(flattened_retrain_states, flattened_FT_states, p=2)
    print(retrain_FT_dist)
    retrain_GA_dist = torch.dist(flattened_retrain_states, flattened_GA_states, p=2)
    print(retrain_GA_dist)
    orig_model_retrain_dist = torch.dist(flattened_orig_model_states, flattened_retrain_states, p=2)
    print(orig_model_retrain_dist)
    orig_model_FT_dist = torch.dist(flattened_orig_model_states, flattened_FT_states, p=2)
    print(orig_model_FT_dist)
    orig_model_GA_dist = torch.dist(flattened_orig_model_states, flattened_GA_states, p=2)
    print(orig_model_GA_dist)

    final_dict['retrain_FT_L2_dist'] = retrain_FT_dist
    final_dict['retrain_GA_L2_dist'] = retrain_GA_dist
    final_dict['orig_model_retrain_L2_dist'] = orig_model_retrain_dist
    final_dict['orig_model_FT_L2_dist'] = orig_model_FT_dist
    final_dict['orig_model_GA_L2_dist'] = orig_model_GA_dist

    #KL Divergence between weight distributions
    retrain_FT_KL = get_kl_divergence(retrain_hist, FT_hist)
    print(retrain_FT_KL)
    retrain_GA_KL = get_kl_divergence(retrain_hist, GA_hist)
    print(retrain_GA_KL)
    orig_model_retrain_KL = get_kl_divergence(orig_model_hist, retrain_hist)
    print(orig_model_retrain_KL)
    orig_model_FT_KL = get_kl_divergence(orig_model_hist, FT_hist)
    print(orig_model_FT_KL)
    orig_model_GA_KL = get_kl_divergence(orig_model_hist, GA_hist)
    print(orig_model_GA_KL)

    final_dict['retrain_FT_KL'] = retrain_FT_KL
    final_dict['retrain_GA_KL'] = retrain_GA_KL
    final_dict['orig_model_retrain_KL'] = orig_model_retrain_KL
    final_dict['orig_model_FT_KL'] = orig_model_FT_KL
    final_dict['orig_model_GA_KL'] = orig_model_GA_KL

    ###OUTPUT SPACE###
    #Wasserstein Distance on test dataset
    w_test_distance_RT_FT = get_wasserstein_distance(retrain_pred, FT_pred)
    print(w_test_distance_RT_FT)
    w_test_distance_RT_GA = get_wasserstein_distance(retrain_pred, GA_pred)
    print(w_test_distance_RT_GA)
    w_test_distance_O_RT = get_wasserstein_distance(orig_model_pred, retrain_pred)
    print(w_test_distance_O_RT)
    w_test_distance_O_FT = get_wasserstein_distance(orig_model_pred, FT_pred)
    print(w_test_distance_O_FT)
    w_test_distance_O_GA = get_wasserstein_distance(orig_model_pred, GA_pred)
    print(w_test_distance_O_GA)

    final_dict['w_test_distance_RT_FT'] = w_test_distance_RT_FT
    final_dict['w_test_distance_RT_GA'] = w_test_distance_RT_GA
    final_dict['w_test_distance_O_RT'] = w_test_distance_O_RT
    final_dict['w_test_distance_O_FT'] = w_test_distance_O_FT
    final_dict['w_test_distance_O_GA'] = w_test_distance_O_GA

    #Wasserstein Distance on forget dataset
    w_forget_distance_RT_FT = get_wasserstein_distance(retrain_forget_pred, FT_forget_pred)
    print(w_forget_distance_RT_FT)
    w_forget_distance_RT_GA = get_wasserstein_distance(retrain_forget_pred, GA_forget_pred)
    print(w_forget_distance_RT_GA)
    w_forget_distance_O_RT = get_wasserstein_distance(orig_forget_pred, retrain_forget_pred)
    print(w_forget_distance_O_RT)
    w_forget_distance_O_FT = get_wasserstein_distance(orig_forget_pred, FT_forget_pred)
    print(w_forget_distance_O_FT)
    w_forget_distance_O_GA = get_wasserstein_distance(orig_forget_pred, GA_forget_pred)
    print(w_forget_distance_O_GA)

    final_dict['w_forget_distance_RT_FT'] = w_forget_distance_RT_FT
    final_dict['w_forget_distance_RT_GA'] = w_forget_distance_RT_GA
    final_dict['w_forget_distance_O_RT'] = w_forget_distance_O_RT
    final_dict['w_forget_distance_O_FT'] = w_forget_distance_O_FT
    final_dict['w_forget_distance_O_GA'] = w_forget_distance_O_GA

    torch.save(final_dict, 'all_metrics.json')
