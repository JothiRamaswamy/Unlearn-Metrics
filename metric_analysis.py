import json
import matplotlib.pyplot as plt

with open('retrain_metrics.json', 'r') as retrainfile:
    retrain_metrics = json.loads(json.load(retrainfile))
    retrain_accuracy = retrain_metrics['accuracy']
    retrain_MIA_efficacy = retrain_metrics['SVC_MIA_forget_efficacy']
    retrain_MIA_privacy = retrain_metrics['SVC_MIA_training_privacy']
    
with open('FT_metrics.json', 'r') as FTfile:
    FT_metrics = json.loads(json.load(FTfile))
    FT_accuracy = FT_metrics['accuracy']
    FT_MIA_efficacy = FT_metrics['SVC_MIA_forget_efficacy']
    FT_MIA_privacy = FT_metrics['SVC_MIA_training_privacy']
    
with open('GA_metrics.json', 'r') as GAfile:
    GA_metrics = json.loads(json.load(GAfile)) 
    GA_accuracy = GA_metrics['accuracy'] 
    GA_MIA_efficacy = GA_metrics['SVC_MIA_forget_efficacy']
    GA_MIA_privacy = GA_metrics['SVC_MIA_training_privacy']
    

def plot_all_accuracies():
    colors = ['blue', 'green', 'red']
    markers = {'retain': 'o', 'forget': 's', 'val': '^', 'test': 'x'} 

    plt.figure(figsize=(10, 6))

    for i, metrics in enumerate([retrain_accuracy, FT_accuracy, GA_accuracy]):
        for j, (acc_type, value) in enumerate(metrics.items()):
            plt.scatter(j, value, color=colors[i], marker=markers[acc_type], s=100, label=f"{acc_type} - {'Retrain' if i==0 else 'FT' if i==1 else 'GA'}" if j==0 else "")

    # Add legend, title, and labels
    plt.legend()
    plt.xticks(range(len(markers)), labels=markers.keys())
    plt.xlabel('Accuracy Type')
    plt.ylabel('Accuracy Value (%)')
    plt.title('Comparison of Accuracy Types Across Different Metrics')

    # Show the plot
    plt.savefig('accuracies.pdf')
    plt.show()
    
def plot_accuracies_MIA_efficacy():
    efficacies = [retrain_MIA_efficacy, FT_MIA_efficacy, GA_MIA_efficacy]
    markers = ['o', 's', '^']
    colors = {'retain': 'green', 'forget': 'blue', 'val': 'red', 'test': 'purple'} 

    plt.figure(figsize=(10, 6))

    for i, metrics in enumerate([retrain_accuracy, FT_accuracy, GA_accuracy]):
        for j, (acc_type, value) in enumerate(metrics.items()):
            plt.scatter(efficacies[i]['correctness'], value, color=colors[acc_type], marker=markers[i], s=100, label=f"{acc_type} - {'Retrain' if i==0 else 'FT' if i==1 else 'GA'}" if j==0 else "")

    # Add legend, title, and labels
    plt.xlabel('MIA efficiacy correctness')
    plt.ylabel('Accuracy Value (%)')
    plt.title('Comparison of Accuracy Types Across Different Metrics')

    # Show the plot
    plt.savefig('accuracy_efficacy.pdf')
    
def plot_accuracies_MIA_privacy():
    privacies = [retrain_MIA_privacy, FT_MIA_privacy, GA_MIA_privacy]
    markers = ['o', 's', '^']
    colors = {'retain': 'green', 'forget': 'blue', 'val': 'red', 'test': 'purple'} 

    plt.figure(figsize=(10, 6))

    for i, metrics in enumerate([retrain_accuracy, FT_accuracy, GA_accuracy]):
        for j, (acc_type, value) in enumerate(metrics.items()):
            plt.scatter(privacies[i]['correctness'], value, color=colors[acc_type], marker=markers[i], s=100, label=f"{acc_type} - {'Retrain' if i==0 else 'FT' if i==1 else 'GA'}" if j==0 else "")

    # Add legend, title, and labels
    plt.xlabel('MIA privacy correctness')
    plt.ylabel('Accuracy Value (%)')
    plt.title('Comparison of Accuracy Types Across Different Metrics')

    # Show the plot
    plt.savefig('accuracy_privacy.pdf')
    
def plot_forget_MIA_efficacy():
    efficacies = [retrain_MIA_efficacy, FT_MIA_efficacy, GA_MIA_efficacy]
    markers = ['o', 's', '^']
    
    plt.figure(figsize=(10, 6))

    for i, metrics in enumerate([retrain_accuracy, FT_accuracy, GA_accuracy]):
        plt.scatter(efficacies[i]['correctness'], metrics['forget'], color='blue', marker=markers[i], s=100)

    # Add legend, title, and labels
    plt.xlabel('MIA efficacy correctness')
    plt.ylabel('Forget Accuracy Value (%)')
    plt.title('Comparison of Accuracy Types Across Different Metrics')

    # Show the plot
    plt.savefig('forget_efficacy.pdf')
    
def plot_retain_test():
    markers = ['o', 's', '^']
    
    plt.figure(figsize=(10, 6))

    for i, metrics in enumerate([retrain_accuracy, FT_accuracy, GA_accuracy]):
        plt.scatter(metrics['retain'], metrics['test'], color='blue', marker=markers[i], s=100)

    # Add legend, title, and labels
    plt.xlabel('Retain Accuracy Value (%)')
    plt.ylabel('Test Accuracy Value (%)')
    plt.title('Comparison of Accuracy Types Across Different Metrics')

    # Show the plot
    plt.savefig('retain_test.pdf')


plot_accuracies_MIA_efficacy()
plot_accuracies_MIA_privacy()
plot_forget_MIA_efficacy()
plot_retain_test()
print(retrain_metrics)


