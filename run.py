import torch
import numpy as np
import pickle
from utils import load_mnist, load_cifar
from model import create_student, create_teacher
from train import kd_train

from datetime import datetime


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

    run_experiment_alpha_temperature('cifar', torch_device)
