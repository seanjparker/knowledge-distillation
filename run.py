import torch
from torch.utils.data import Subset
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from utils import load_mnist, load_cifar, calc_accuracy
from model import create_student, create_teacher, create_assistant
from train import kd_train, train_model

from datetime import datetime


def train_student_from_teacher_assistant(dataset, device):
    # Load dataset
    train_dataset, test_dataset = load_mnist() if dataset == 'mnist' else load_cifar()
    in_dims = 3 if dataset == 'cifar' else 1

    assistant_model = create_assistant(device, 10, in_dims)

    # Train with custom args
    assistant_model = train_model((train_dataset, test_dataset), assistant_model, device=device)

    # Save trained model with name
    timestamp = datetime.now().strftime("%H:%M:%S")
    torch.save(assistant_model.state_dict(), f'./models/{timestamp}_assistant_{dataset}.pt')


def train_teacher(dataset, device):
    # Load dataset
    train_dataset, test_dataset = load_mnist() if dataset == 'mnist' else load_cifar()
    in_dims = 3 if dataset == 'cifar' else 1

    teacher_model = create_teacher(device, 1, in_dims=in_dims)

    # Train with custom args
    teacher_model = train_model((train_dataset, test_dataset), teacher_model, device=device)

    # Save trained model with name
    timestamp = datetime.now().strftime("%H:%M:%S")
    torch.save(teacher_model.state_dict(), f'./models/{timestamp}_teacher_{dataset}.pt')


def run_experiment_small_dataset(dataset, device, how_much, temperature=10, alpha=0.1):
    # Load dataset
    train_dataset, test_dataset = load_mnist() if dataset == 'mnist' else load_cifar()

    # Create a small subset of the original training dataset, with approx. equal distribution of classes
    indices = np.arange(len(train_dataset.dataset))
    number_train_samples = int(len(train_dataset.dataset) * how_much)
    train_indices, _ = train_test_split(indices,
                                        train_size=number_train_samples, stratify=train_dataset.dataset.targets)
    train_subset = torch.utils.data.Subset(train_dataset.dataset, train_indices)
    small_train_dataset = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=2)

    # Create teacher & student models
    student_model = create_student(device, temperature)
    teacher_model = create_teacher(device, temperature, './models/cifar_teacher_0.pt')

    student_model, logged_data = kd_train((small_train_dataset, test_dataset),
                                          teacher_model, student_model,
                                          temperature=temperature, alpha=alpha, device=device)

    # Save trained model with name
    timestamp = datetime.now().strftime("%H:%M:%S")
    filename = f'{timestamp}_student_small_dataset'
    torch.save(student_model.state_dict(), f'./models/{filename}.pt')

    student_model.eval()
    print(calc_accuracy(test_dataset, student_model, device))
    student_model.train()


def run_experiment_alpha_temperature(dataset, device):
    # Load dataset
    train_dataset, test_dataset = load_mnist() if dataset == 'mnist' else load_cifar()

    kd_temperature = [2, 5, 7, 10, 15, 20]
    kd_alpha = [0.1, 0.5, 0.99]
    for i in range(len(kd_temperature)):
        for j in range(len(kd_alpha)):
            # Create teacher & student models
            student_model = create_student(device, kd_temperature[i])
            teacher_model = create_teacher(device, kd_temperature[i], './models/cifar_teacher_0.pt')

            # Train with custom args
            student_model, logged_data = kd_train((train_dataset, test_dataset),
                                                  teacher_model, student_model,
                                                  temperature=kd_temperature[i], alpha=kd_alpha[j], device=device)

            # Save trained model with name
            timestamp = datetime.now().strftime("%H:%M:%S")
            filename = f'{timestamp}_student_t{kd_temperature[i]}_a{kd_alpha[j]:.2f}'.replace('.', '-')
            torch.save(student_model.state_dict(), f'./models/{filename}.pt')

            # Save results
            with open(f'{filename}.pickle', 'wb') as handle:
                pickle.dump(logged_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Sets the device for training, make sure we are using CUDA when training
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name() if torch_device.type == 'cuda' else 'cpu'
    print(f'Using device: {device_name}, type {torch_device}')

    train_student_from_teacher_assistant('cifar', torch_device)
