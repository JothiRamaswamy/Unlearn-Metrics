from dataset import *
from utils import *
from models import *
from pruner import *
import json
import torchvision
import torchvision.transforms as transforms

def save_models(directory, seed=2):
    MODEL_FILE_INPUTS = {'FTcheckpoint.pth.tar': 'FT_model.pt', 'FTeval_result.pth.tar': 'FT_metrics.json', 'GAcheckpoint.pth.tar':'GA_model.pt', 'GAeval_result.pth.tar': 'GA_metrics.json', 'retraincheckpoint.pth.tar': 'retrain_model.pt', 'retraineval_result.pth.tar': 'retrain_metrics.json'}

    for input, output in MODEL_FILE_INPUTS.items():
        model = torch.load(directory + input, map_location=torch.device('cpu'))
        if '.pt' in output:
            torch.save(model, directory + output)
        else:
            model = json.dumps(model)
            torch.save(model, directory + output)

    get_all_pred_logits(directory, seed)

def get_model(filename):
    model_data = torch.load(filename)
    model_state_dict = model_data['state_dict']
    model = load_model(model_state_dict)
    model.load_state_dict(model_state_dict)
    return model

def get_all_pred_logits(directory, seed):
    MODEL_FILE_INPUTS = {'model.pt': ['predictions/orig_test_pred_logits.pt', 'predictions/orig_forget_pred_logits.pt'],
                         'retrain_model.pt': ['predictions/retrain_test_pred_logits.pt', 'predictions/retrain_forget_pred_logits.pt'],
                         'FT_model.pt': ['predictions/FT_test_pred_logits.pt', 'predictions/FT_forget_pred_logits.pt'],
                         'GA_model.pt': ['predictions/GA_test_pred_logits.pt', 'predictions/GA_forget_pred_logits.pt']}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the data
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    forgetset = get_forget_set(seed)
    for input, outputs in MODEL_FILE_INPUTS.items():
        model = get_model(directory + input)
        get_predictions(model, testset, directory + outputs[0])
        get_predictions(model, forgetset, directory + outputs[1])

def get_forget_set(seed=2):
    def replace_loader_dataset(
        dataset, batch_size=256, seed=2, shuffle=True
    ):
        setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    loaders = cifar10_dataloaders(
            batch_size=256,
            data_dir="datasets/cifar10",
            num_workers=8,
            class_to_replace=0,
            num_indexes_to_replace=4500,
            indexes_to_replace=None,
            seed=seed,
            only_mark=True,
            shuffle=True,
            no_aug=False,
        )
    forget_dataset = copy.deepcopy(loaders[0].dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(
        forget_dataset, seed=seed, shuffle=True
    )
    print(len(forget_dataset))
    retain_dataset = copy.deepcopy(loaders[0].dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(
        retain_dataset, seed=seed, shuffle=True
    )
    assert(len(forget_dataset) + len(retain_dataset) == len(loaders[0].dataset))
    return forget_dataset

def get_predictions(model, testset, filename):

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    predictions = []
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data
            outputs = model(inputs)
            predictions.append(torch.tensor(outputs))

    predictions_tensor = torch.cat(predictions, dim=0)
    torch.save(predictions_tensor, filename)
    return predictions_tensor

def load_model(state_dict):

    args_seed = 2
    args_train_seed = 1
    args_arch = 'resnet18'

    classes = 10
    normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )

    if args_train_seed is None:
        args_train_seed = args_seed
    setup_seed(args_train_seed)

    model = model_dict[args_arch](num_classes=classes)

    setup_seed(args_train_seed)

    model.normalize = normalization

    mask = extract_mask(state_dict)
    prune_model_custom(model, mask)

    return model

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)
