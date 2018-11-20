import numpy as np
import random
sku = 50
matrix = np.loadtxt("matrix_adaptive/m.txt",  delimiter = "\t")
#matrix = np.loadtxt("confusion_matrix.txt",  delimiter = " ")
matrix = matrix[:sku, :sku]
matrix[matrix < 4] = 0
matrix = matrix.astype(int)
print("The size of matrix is", matrix.shape)

train_list = [line.strip() for line in open("trainSample50.txt")]
batch_size = 100
sample_num = len(train_list)
n = sample_num // batch_size
value_list = []
for i in range(n):
    label_list = []
    for j in range(batch_size):
        label = train_list[i * batch_size + j].split()[-1]
        label = int(label)
        if label not in label_list:
            label_list.append(label)
    value = 0
    for k in label_list:
        for t in label_list:
            if k != t:
                value += matrix[k][t]
    value_list.append(value)
value_list = np.array(value_list)
np.savetxt("confusionValue.txt", value_list, fmt = "%d", delimiter = "\n")
print("We get the confusion values")

print("Regenerate the training list")
new_list = open("newSample50_online.txt", "w")
high = value_list.sum()
low = 0
rand_list = np.random.randint(low = low, high = high, size = 10000)
print(len(rand_list))
cnt = 0
for num in rand_list:
    amount = 0
    index = 0
    for i in range(len(value_list)):
        amount +=value_list[i]
        if amount > num:
            index = i
            break
    batch = train_list[index * batch_size : index * batch_size + batch_size]
    if len(batch)!= 100:
        print("What's wrong")
    for i in range(100):
        new_list.writelines(batch[i] + "\n")
        cnt += 1
print(cnt)
print("We regenerated a new list in nweTrain.txt")
    
