import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import res_single as res
import time
import argparse
from PIL import Image
from load import dataLoader, dataAugmentation

class modelEval():
    def __init__(self, model, state_dict, class_num):
        self.model = model
        self.weight = state_dict
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_num = class_num
        self.confusion_matrix = np.zeros([class_num, class_num])
    def loadWeight(self):
        print("Model file:" + self.weight)
        model_dict = model.state_dict()
        weight = torch.load(self.weight)
        trained_dict = {k: v for k, v in weight.items() if k in model_dict}
        self.model.load_state_dict(trained_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
    def loadData(self, dataPath, dataList, dataset = ''):
        augmentation = dataAugmentation()
        data = dataLoader(dataPath, dataList, dataset = dataset, data_transforms = augmentation.data_transforms)
        return data
    def singleSearchL2(self):
        rightCnt = 0
        t0 = time.time()
        counter = np.zeros(self.class_num)
        correct_counter = np.zeros(self.class_num)
        for i in range(len(self.test)):
            label = self.testLabel[i]
            counter[int(label)] += 1
            feature = self.test[i]
            dist = np.sum(np.square(self.lib - feature), axis = 1)
            index = dist.argsort(axis = 0)[0]
            top1 = self.libLabel[int(index)]
            if top1 == label:
                rightCnt += 1
                correct_counter[int(label)] += 1
            else:
                self.confusion_matrix[int(label)][int(top1)] += 1
        t1 = time.time()
        recall = correct_counter / counter
        mean_recall = np.mean(recall[0:50])
        precision = rightCnt / len(self.test)
        results = open("model_results.txt", "a")
        results.writelines(self.weight + "\t" + str(mean_recall) + "\t" + str(precision) + "\n") 
        np.savetxt("recall.txt", recall, fmt = "%.4f", delimiter = "\n")
        np.savetxt("confusion_matrix.txt", self.confusion_matrix, fmt = "%d", delimiter = "\t")
        print("Average time is %f" % ((t1 - t0) / len(self.test)))
        print("Precision is %f" % (rightCnt / len(self.test)))
        print("Recall is \n", recall)
        print("Mean recall is ", (np.mean(recall[0:50])))
        
    def getMatrix(self, path, listName, fileName, labelName):
        data = self.loadData(path, listName, 'testImages')
        dataloaders = torch.utils.data.DataLoader(data, batch_size = 1, shuffle = False)
        t = 0.0
        matrix = []
        labels = []
        cnt = 0
        for image, label in dataloaders:
            cnt += 1
            with torch.no_grad():
                image = image.to(self.device)
                t0 = time.time()
                x = model(image, label)
                t1 = time.time()
                t += t1 - t0
                features = x.to("cpu")
                f = features.numpy()
                matrix.append(f[0])
                labels.append(label)
                if cnt % 10000 == 0:
                    print("%d images" % cnt)
        print("All %d images" % cnt)
        matrix = np.stack(matrix, axis = 0)
        labels = np.array(labels)
        np.savetxt(fileName, matrix, fmt = "%f", delimiter = " ")
        np.savetxt(labelName, labels, fmt = "%d", delimiter = "\n")
        print("There are %d images. Average time of each iteration is %f" % (cnt, t / cnt))
        return matrix, labels
    def loadMatrix(dataset = "dataset.txt", dataset_label = "dataset_label.txt", testset = "testset.txt", testset_label = "testset_label.txt"):
        print("Load saved features and labels.")
        self.lib = np.loadtxt(dataset)
        self.libLabel = np.loadtxt(dataset_label)
        self.test = np.loadtxt(testset)
        self.testLabel = np.loadtxt(testset_label)
        
    def evaluate(self, searchPath, searchList, testPath, testList, isTrain = 1, rerun = 0):
        if not rerun:
            print("Start loading model")
            self.loadWeight()
            print("Load model successfully.")
            
            if isTrain:
                print("Start building search pool")
                self.lib, self.libLabel = self.getMatrix(searchPath, searchList, "dataset.txt", "dataset_label.txt")
                print("Build search pool successfully.")
        
            print("Start test images")
            self.test, self.testLabel = self.getMatrix(testPath, testList, "testset.txt", "testset_label.txt")
            print("All features of test images are saved.")
        else:
            print("Start rerun with saved matrix.")
            self.loadMatrix()
        print("Start L2 searching")
        self.singleSearchL2()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "test parameters")
    parser.add_argument("-weights", dest = "weights", type = str, default = "model/resnet_Asoftmax_iter_100000.pt")
    parser.add_argument("-class_num", dest = "class_num", type = int, default = 461)
    parser.add_argument("-searchList", dest = "searchList", type = str, default = "searchList.txt")
    parser.add_argument("-searchPath", dest = "searchPath", type = str, default = "/export/home/dyh/workspace/circle_k/for_douyuhao/data/searchImages/")
    parser.add_argument("-testList", dest = "testList", type = str, default = "testList.txt")
    parser.add_argument("-testPath", dest = "testPath", type = str, default = "/export/home/dyh/workspace/circlr_k/for_douyuhao/data/testImages/")
    parser.add_argument("-isTrain", dest = "isTrain", type = int, default = 1)
    parser.add_argument("-rerun", dest = "rerun", type = int, default = 0)
    args = parser.parse_args()
    model = res.resnet18(pretrained = False, num_classes = args.class_num)
    #model = nn.DataParallel(model, [0, 1, 2, 3])
    instance = modelEval(model, args.weights , class_num = args.class_num)
    instance.evaluate("../data/searchImages/", "searchList50.txt", "../data/testImages/", "testList50.txt", isTrain = args.isTrain, rerun = args.rerun)

