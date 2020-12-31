import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import numpy as np
import time
from typing import Tuple

from utils import weight_reset


def kd_train(datasets: Tuple[DataLoader, DataLoader], teacher, student, student_optimizer=None,
             lr=0.001, epochs=30, alpha=0.2, temperature=5, device='cpu'):
    # Load the datasets, train and test (reports test accuracy every 10 epochs)
    train, test = datasets[0], datasets[1]

    # Reset the student network parameters before training
    # Sets up the optimizer
    student.apply(weight_reset)
    student_optimizer = student_optimizer \
        if student_optimizer is not None else torch.optim.Adam(student.parameters(), lr=lr)

    # Set up the lr scheduler and loss functions
    student_lr_sch = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, [10, 25, 40], gamma=0.5)
    student_loss_fn = nn.CrossEntropyLoss()
    distillation_loss_fn = torch.nn.KLDivLoss(reduction='mean')

    # Setup arrays to record data while training
    losses = np.zeros((epochs,), dtype=np.float32)
    train_correct = np.zeros((epochs,), dtype=np.float32)
    for ep in range(epochs):
        start_time = time.time()
        for i, (imgs, labels) in enumerate(train):
            # Move the images and labels to the device
            imgs, labels = imgs.to(device), labels.to(device)

            student_optimizer.zero_grad()

            # Forward pass of the teacher with input
            with torch.no_grad():
                teacher_output = teacher(imgs).to(device)
            # Forward pass of the student
            student_output = student(imgs).to(device)

            # Calculate loss, soft target loss
            # https://arxiv.org/pdf/1503.02531.pdf
            student_loss = student_loss_fn(student_output, labels)
            distill_loss = distillation_loss_fn(teacher_output, student_output)

            # Magnitudes of gradients scale with 1/T^2 --> multiply loss by T^2
            loss = (alpha * student_loss + (1 - alpha) * distill_loss) * temperature * temperature

            # Backprop for student model only
            loss.backward()
            student_optimizer.step()

            # Calculate train accuracy
            pred_class = torch.argmax(student_output, dim=1).to(device)
            correct = torch.eq(pred_class, labels).sum()

            # Record relevant quantities for minibatch of current epoch
            train_correct[ep] += float(correct)
            losses[ep] += loss.detach().item()

        # Step LR scheduler at the end of every epoch
        student_lr_sch.step()

        print(f'epoch: {ep + 1}, loss: {losses[ep] / len(train.dataset):.2e}, train acc: {train_correct[ep] / len(train.dataset):.3f}, time: {(time.time() - start_time):.2f}s')

    losses = losses / len(train.dataset)
    train_correct = train_correct / len(train.dataset)
    records = {'train_acc': train_correct, 'loss': losses}
    return student, records
