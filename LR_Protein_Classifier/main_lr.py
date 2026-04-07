import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

class LRModel:
    # todo:
    """
        Initialize Logistic Regression (from sklearn) model.

    """
    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """
    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def __init__(self):
        pass

    def train(self, train_data, train_targets):
        pass

    def evaluate(self, data, targets):
        pass


class LRFromScratch:
    # todo:
    def __init__(self):
        pass

    def train(self, train_data, train_targets):
        pass
    
    def evaluate(self, data, targets):
        pass


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
    

    model = LRModel()
    # model = LRFromScratch()
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

