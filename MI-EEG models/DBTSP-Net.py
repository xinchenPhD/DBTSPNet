import argparse
import os
import sys
sys.path.append('./')
# os.environ["OMP_NUM_THREADS"] = "0"

from multiprocessing import Pool
import numpy as np
import random
import datetime
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import scipy.io
from torch.backends import cudnn
import optuna
import matplotlib.pyplot as plt
# import torch, gc
#
# gc.collect()
# torch.cuda.empty_cache()


cudnn.benchmark = False
cudnn.deterministic = True

from EEG_model.DuralSP_weight import DuralSP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class ExP():
    def __init__(self, args, nsub, lr, b1, b2, weight_transformer, weight_bilstm, batch_size):
        super(ExP, self).__init__()
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.c_dim = 4
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.weight_transformer = weight_transformer
        self.weight_bilstm = weight_bilstm
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = args.data_root

        self.exp_name = args.exp_name
        self.results_dir = os.path.join(args.results_dir, self.exp_name)
        create_dir_if_not_exists(self.results_dir)
        self.log_write = open(os.path.join(self.results_dir, f"log_subject{self.nSub}.txt"), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = DuralSP(
            emb_size=64, depth=6, n_classes=4,
            weight_transformer=self.weight_transformer,
            weight_bilstm=self.weight_bilstm
        ).cuda()

        gpus = [i for i in range(torch.cuda.device_count())]
        self.model = nn.DataParallel(self.model, device_ids=gpus).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label - 1).cuda().long()
        return aug_data, aug_label

    def get_source_data(self):
        # Train data
        self.total_data = scipy.io.loadmat(self.root + f'A0{self.nSub}T.mat')
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # Test data
        self.test_tmp = scipy.io.loadmat(self.root + f'A0{self.nSub}E.mat')
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # Standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Adam Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2), weight_decay=1e-4) #TODO
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.937, weight_decay=1e-2)    ADAMW

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0

        for e in range(self.n_epochs):
            self.model.train()

            epoch_train_loss = 0
            epoch_train_correct = 0
            epoch_train_total = 0

            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # Data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                # print("img.shape:", img.shape)
                # model_output = self.model(img)
                # print("------", {model_output})

                tok, outputs = self.model(img)  # TODO

                # print("-----", outputs.shape)
                # print(tok.shape, outputs.shpae)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_correct += (torch.max(outputs, 1)[1] == label).sum().item()
                epoch_train_total += label.size(0)
            acc = epoch_train_correct / epoch_train_total
            epoch_train_loss /= len(self.dataloader)

            if (e + 1) % 1 == 0:
                self.model.eval()
            with torch.no_grad():  #####
                Tok, Cls = self.model(test_data)
                # print('******************',Tok.shape, Cls.shape)
                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                # self.train_losses.append(epoch_train_loss)
                # self.test_losses.append(loss_test.item())
                # self.train_accuracies.append(acc)
                # self.test_accuracies.append(train_acc)

                self.train_losses.append(epoch_train_loss)
                self.test_losses.append(loss_test.item())
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(acc)

                print(f'Epoch: {e}, Train loss: {loss.item():.6f}, Test loss: {loss_test.item():.6f}, Train accuracy: {train_acc:.6f}, Test accuracy: {acc:.6f}')
                self.log_write.write(f'{e}    {acc}\n')
                num += 1
                averAcc += acc
                if acc > bestAcc:
                    bestAcc = acc

        # torch.save(self.model.module.state_dict(), os.path.join(self.results_dir, 'model.pth'))
        torch.save(self.model.module, os.path.join(self.results_dir, 'model.pth'))
        averAcc /= num
        print(f'The average accuracy is: {averAcc}')
        print(f'The best accuracy is: {bestAcc}')
        self.log_write.write(f'The average accuracy is: {averAcc}\n')
        self.log_write.write(f'The best accuracy is: {bestAcc}\n')

        # self.plot_metrics()
        return bestAcc, averAcc

    # def plot_metrics(self):
    #     epochs = range(1, len(self.train_losses) + 1)
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
    #     plt.plot(epochs, self.test_losses, 'r-', label='Test Loss')
    #     plt.title('Loss Curve')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.results_dir, 'loss_curve.png'))
    #     plt.show()
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
    #     plt.plot(epochs, self.test_accuracies, 'r-', label='Test Accuracy')
    #     plt.title('Accuracy Curve')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.results_dir, 'accuracy_curve.png'))
    #     plt.show()


def objective(trial, args, nsub):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    b1 = trial.suggest_float('b1', 0.4, 0.9, log=True)
    b2 = trial.suggest_float('b2', 0.8, 0.999, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    weight_transformer = trial.suggest_float('weight_transformer', 0.1, 0.9, log=True)
    weight_bilstm = trial.suggest_float('weight_bilstm', 0.1, 0.9, log=True)

    exp = ExP(args, nsub=nsub, lr=lr, b1=b1, b2=b2, batch_size=batch_size,
              weight_transformer=weight_transformer, weight_bilstm=weight_bilstm)

    bestAcc, averAcc = exp.train()

    return bestAcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='E:\\model_updating-------\\EEGNet\\EEG\\standard_BCICIV_2a_data\\', help='Root directory of the data')
    parser.add_argument('--results_dir', type=str, default='Results', help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default='victory2', help='Experiment name for saving results')
    parser.add_argument('--batch_size', type=int, default=72, help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    args = parser.parse_args()

    best = 0
    aver = 0
    result_write_path = os.path.join(args.results_dir, args.exp_name, "sub_result.txt")
    create_dir_if_not_exists(os.path.dirname(result_write_path))
    result_write = open(result_write_path, "w")

    for nsub in range(9, 10):
        print(f"---------------------Running trials for-------------- Subject {nsub}...")

        # 创建 Optuna study并优化超参数
        study = optuna.create_study(direction='maximize')
        # study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, args, nsub), n_trials=15)

        print(f"Best hyperparameters for Subject {nsub}: {study.best_params}")
        print(f"Best accuracy for Subject {nsub}: {study.best_value}")

        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2021)
        print(f'Seed is {seed_n}')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print("使用优化后的超参数训练")
        exp = ExP(args, nsub=nsub, lr=study.best_params['lr'], b1=study.best_params['b1'],
                  b2=study.best_params['b2'], batch_size=study.best_params['batch_size'],
                  weight_transformer=study.best_params['weight_transformer'],
                  weight_bilstm=study.best_params['weight_bilstm'])

        bestAcc, averAcc = exp.train()
        print(f'THE BEST ACCURACY  from training IS {bestAcc}')

        # final_best_acc = max(study.best_value, bestAcc)

        # result_write.write(f'Subject {nsub} : Seed is: {seed_n}\n')
        # result_write.write(f'Subject {nsub} : Best accuracy from Optuna: {study.best_value}\n')
        # result_write.write(f'Subject {nsub} : Best accuracy from training: {bestAcc}\n')
        # result_write.write(f'Subject {nsub} : Final best accuracy: {final_best_acc}\n')
        # result_write.write(f'Subject {nsub} : The average accuracy is: {averAcc}\n')

        result_write.write(f'Subject {nsub} : Seed is: {seed_n}\n')
        # result_write.write(f'Subject {nsub} : The best accuracy is: {bestAcc}\n')
        result_write.write(f'Subject {nsub} : The best accuracy is: {study.best_value}\n')
        result_write.write(f'Subject {nsub} : The average accuracy is: {averAcc}\n')

        endtime = datetime.datetime.now()
        print(f'subject {nsub} duration: {str(endtime - starttime)}')
        # best += bestAcc
        best += study.best_value
        aver += averAcc

    best /= 9
    aver /= 9

    result_write.write(f'**The average Best accuracy is: {best}\n')
    result_write.write(f'The average Aver accuracy is: {aver}\n')
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()