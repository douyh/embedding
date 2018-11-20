import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
import res_single as res
from torch.utils.data import Dataset
from PIL import Image
import fast_pair_loss
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image", path)
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms = None, loader = default_loader):
        self.image = [os.path.join(img_path, line.strip().split()[0]) for line in open(txt_path)]
        self.label = [int(line.strip().split()[1]) for line in open(txt_path)]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        imageName = self.image[item]
        label = self.label[item]
        image = self.loader(imageName)
        if self.data_transforms is not None:
            try:
                image = self.data_transforms[self.dataset](image)
            except:
                print("Cannot transform image ", imageName)
        return image, label

class Net():
    
    def __init__(self, data_dir, train_file, test_file, class_num):
        # data augmentation
        self.data_transforms = {
            "trainImages": transforms.Compose([ 
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
            "testImages": transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        }

        # load data
        self.data_dir = data_dir
        self.train_set = customData(img_path = self.data_dir + "/trainImages/", txt_path = train_file, data_transforms = self.data_transforms, dataset = "trainImages")
        self.train_data = torch.utils.data.DataLoader(self.train_set, batch_size = 100, shuffle = False, num_workers = 16)
        self.test_set = customData(img_path = self.data_dir + "/testImages/", txt_path = test_file, data_transforms = self.data_transforms, dataset = "testImages")
        self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size = 100, shuffle = False, num_workers = 16)
        
        self.class_num = class_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def train(self, model, loss_function, optimizer, scheduler, model_dir, matrix_dir, start = 0, test_step = 1000, save_step = 10000, num_epochs = 25):
        print("==" * 50)
        print("Training starts")
        cnt = start
        confusion_matrix = torch.zeros([self.class_num, self.class_num]).cuda()
        counter_matrix = torch.zeros([self.class_num, self.class_num]).cuda()
        for epoch in range(num_epochs):
            trainLoss = 0.0
            trainDataIterator = self.train_data
            testDataIterator = self.test_data

            model.train()
            interval = 0
            for inputs, labels in trainDataIterator:
                t0 = time.time()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                cnt += 1
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    x = model(inputs, labels)
                    loss, index, avg_loss = loss_function(x, labels, hard_ratio = 1.0)
                    for i in range(10):
                        for j in range(10):
                            s1 = int(index[i][j][0].item())
                            s2 = int(index[i][j][1].item())
                            confusion_matrix[s1][s2] += avg_loss[i][j]
                            counter_matrix[s1][s2] += 1
                    # backward + optimize in training phase
                    loss.backward()
                    #scheduler.step()
                    optimizer.step()
                # statistics
                trainLoss += loss
                t1 = time.time()
                interval += t1 - t0
                if cnt % test_step == 0:
                    testCnt = 0
                    print("--"*30)
                    print("Train time consuming %.1fs" % interval)
                    print("%d iterations, train loss: %.4f" % (cnt, trainLoss / test_step))
                    print("The learning rate is", optimizer.param_groups[-1]['lr'])
                    trainLoss = 0.0
                    interval = 0
                    test_t0 = time.time()
                    model.eval()
                    with torch.no_grad():
                        testLoss = 0
                        for testData, testLabel in testDataIterator:
                            testData = testData.to(self.device)
                            testLabel = testLabel.to(self.device)
                            x = model(testData, testLabel)
                            loss, _, _ = loss_function(x, testLabel, hard_ratio = 1.0)
                            testCnt += 1
                            testLoss += loss
                            if testCnt == 40:
                                test_t1 = time.time()
                                print("Test time comsuming %.1fs" %(test_t1 - test_t0))
                                print("%d iterations, test loss: %.4f" % (cnt, testLoss / 40.0))
                                print("--" * 30)
                                break
                
                if cnt % save_step == 0:
                    print("**"*30)
                    time.sleep(20)
                    print("Save model to hdc_iter_%d.pt" % cnt)
                    torch.save(model.state_dict(), model_dir + "/hdc_iter_%d.pt" % cnt)
                    if start == 0:
                        print("Save online confusion matrix in online_iter_%d.txt" % cnt)
                        save_matrix = torch.zeros([self.class_num, self.class_num]).cuda()
                        save_matrix[counter_matrix > 0] = 100.0 * confusion_matrix[counter_matrix > 0] / counter_matrix[counter_matrix > 0]
                        np.savetxt(matrix_dir + "/online_iter_%d.txt" % cnt, save_matrix.to("cpu").detach().numpy(), fmt = "%d", delimiter = "\t")
                    print("**" * 30)

                if cnt == 10000 or cnt == 20000:
                    optimizer.param_groups[-1]['lr'] *= 0.1
                if cnt == 30000:
                    break

                model.train()

        print('Training Done...')
        return model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-reSample", dest = "reSample", type = int, default = 0)
    parser.add_argument("-class_num", dest = "class_num", type = int, default = 50)
    parser.add_argument("-hard_ratio", dest = "hard_ratio", type = float, default = 1.0)
    parser.add_argument("-weights", dest = "weights", type = str, default = "")
    parser.add_argument("-model_dir", dest = "model_dir", type = str, default = "model_hdc")
    parser.add_argument("-matrix_dir", dest = "matrix_dir", type = str, default = "matrix")
    parser.add_argument("-data_dir", dest = "data_dir", type = str, default = "../data/")
    parser.add_argument("-train_file", dest = "train_file", type = str, default = "trainSample50.txt")
    parser.add_argument("-test_file", dest = "test_file", type = str, default = "testSample50.txt")
    parser.add_argument("-start", dest = "start", type = int, default = 0)
    args = parser.parse_args()

    net = Net(data_dir = args.data_dir, train_file = args.train_file, test_file = args.test_file, class_num = args.class_num)
    model = res.resnet18(pretrained = False, num_classes = args.class_num)
    if args.weights == "":
        pretrained_model = models.resnet18(pretrained = True)
        pretrained_dict = pretrained_model.state_dict()
    else:
        pretrained_dict = torch.load(args.weights) 
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    #model = nn.DataParallel(model, [0, 1, 2, 3]) 
    #model.load_state_dict(torch.load("model_hdc/res18_9930/hdc_iter_10000.pt"))

    model = model.to(net.device)
    loss_function = fast_pair_loss.fast_pair_loss()
    
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0002)
    lr_policy = lr_scheduler.StepLR(optimizer, step_size = 10000, gamma = 0.1)
    
    model = net.train(model, loss_function, optimizer, lr_policy, model_dir = args.model_dir, matrix_dir = args.matrix_dir, start = args.start, save_step = 1000, test_step = 100, num_epochs = 3)



