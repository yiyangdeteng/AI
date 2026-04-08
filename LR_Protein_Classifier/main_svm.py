import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class SVMModel:
    # todo:
    """
        Initialize Support Vector Machine (SVM from sklearn) model.

    """
    """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """
    """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """
    def __init__(self):
        self.model = SVC(max_iter=1000)

    def train(self, train_data, train_targets):
        if len(np.unique(train_targets)) > 1:
            self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        if len(targets) > 0:
            return self.model.score(data, targets)
        return 0.0

class SVMFromScratch:
   class SVMFromScratch:
    """
        手动实现的支持向量机模型 (基于梯度下降和 Hinge Loss)
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # 正则化参数，控制间隔大小
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def train(self, train_data, train_targets):
        num_samples, num_features = train_data.shape
        # 1. 将原始标签转换为 -1 和 1
        y_ = np.where(train_targets <= 0, -1, 1)

        self.weights = np.zeros(num_features)
        self.bias = 0

        # 2. 梯度下降迭代
        for _ in range(self.num_iterations):
            # 计算每个样本当前的“分类置信度”：y_i * (w * x_i + b)
            # 结果 >= 1 说明分类极其正确（处于安全区）
            # 结果 < 1 说明分类错误，
            condition = y_ * (np.dot(train_data, self.weights) + self.bias) >= 1
            
            # 首先初始化梯度（正则项）
            dw = 2 * self.lambda_param * self.weights
            db = 0

            # 找出不符合要求的样本
            incorrect_idx = ~condition
            
            # 如果存在处于危险区的样本，它们就会产生 Hinge Loss，需要叠加梯度
            if np.any(incorrect_idx):
                # 累加危险区样本对 w 的偏导数：-y_i * x_i
                dw -= np.dot(train_data[incorrect_idx].T, y_[incorrect_idx])
                # 累加危险区样本对 b 的偏导数：-y_i
                db -= np.sum(y_[incorrect_idx])

            dw /= num_samples
            db /= num_samples
            # 更新导数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def evaluate(self, data, targets):
        # 计算线性输出 z = w * x + b
        linear_output = np.dot(data, self.weights) + self.bias
        
        # z >= 0 判定为正例 1，否则判定为负例 0
        predictions = (linear_output >= 0).astype(int)
        
        return np.mean(predictions == targets)
    

def data_preprocess():
    diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task] # Get the column for the current task
        train_data = []
        test_data = []
        train_targets = []
        test_targets = []
        ## todo: Try to load data/target
        for i , label in enumerate(task_col):
            features = diagrams[i]
            if label == 1:
                train_data.append(features)
                train_targets.append(1)
            elif label == 2:
                train_data.append(features)
                train_targets.append(0)
            elif label == 3:
                test_data.append(features)
                test_targets.append(1)
            elif label == 4:
                test_data.append(features)
                test_targets.append(0)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_targets = np.array(train_targets)
        test_targets = np.array(test_targets)

        # standardize the data
        # 课外探索：标准化数据可以让数据处于相似的范围内，避免某些特征对模型产生过大的影响
        # 这边采用了StandardScaler进行标准化，它会将数据转换为均值为0，标准差为1的分布。这样可以提高模型的训练效果和收敛速度。
        if len(train_data) > 0:
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))
    
    return data_list, target_list

def main():

    data_list, target_list = data_preprocess()

    task_acc_train = []
    task_acc_test = []
    
    ## Todo:Model Initialization 
    ## You can also consider other different settings

    model = SVMModel()
    # model = SVMFromScratch()
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    main()

