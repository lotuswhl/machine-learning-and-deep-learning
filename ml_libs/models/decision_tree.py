#__*__coding=utf-8__*__
import sys
sys.path.append("/home/lotuswhl/projects/machine learning & deep leanring/")

from ml_libs.utils.data_manipulation import separate_data_on_feature_by_threshold
from ml_libs.utils.data_manipulation import calc_gini_group_score
from ml_libs.utils.data_operation import information_entropy
from ml_libs.utils.data_operation import variance_cacl
import numpy as np

"""
决策树的基本模型,以及分类树和回归树,xgboost tree的定义
思路:
    决策树的根本要素在于根据选择的特征以及该特征中选择的阈值来进行划分,因此递归构建子树的根本要义在于
首先构建节点结构;然后需要对输入样本的每一个特征,再对特征的每一个取值,来进行数据分割,然后计算此时的样本的
杂质率,impurity,或则purity也就是样本的整洁率;通常可以有多种方式选择,比如可以选择使用信息增益;而xgboost
则可以根据自己的特征构建基于其目标函数obj或者说损失函数的度量表达式,以此决策最佳的特征和threshold的分割点.
"""


class TreeNode(object):
    """
    通常来说:我们需要定义树的节点结构
    节点可能是分支,也可能是叶子节点,同时要包含子树,因此,节点结构属性可定义如下
    ---
    feature_index: 该节点判别特征依据
    threshold: 该节点根据特征划分依据
    value: 如果是叶子节点,这里通常会是一个具体的值,分类可能是标签,回归通常是数值
    left_brunch: 左分支
    right_brunch: 右分支
    ---
    """
    def __init__(self, feature_index=None,threshold=None,value=None,left_brunch=None,right_brunch=None):
        super(TreeNode, self).__init__()
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_brunch = left_brunch
        self.right_brunch = right_brunch
        self.value = value




class DecisionTree(object):
    """
    决策树的定义

    参数:
    ---
    minimum_sample: 需分割最小节点样本数,也就是样本数量为此值时,不再进行分割
    max_depth: 树的最大深度
    minimum_impurity: 最低的杂质率,或者说样本的分类结果最佳度量,用于获取全局最优
    is_onehot: 是否是onehot编码的标签
    loss: 对于gradient boosting方法需要的损失函数模型
    ---
    """
    def __init__(self,minimum_sample=2,max_depth=float("inf"),minimum_impurity=1e-7,is_onehot=None,loss=None):
        super(DecisionTree,self).__init__()
        self.minimum_sample = minimum_sample
        self.max_depth = max_depth
        self.minimum_impurity = minimum_impurity
        self.is_onehot = is_onehot
        self.loss = loss
        # 添加一个根节点
        self.root = None
        # 添加impurity的计算函数接口
        self._impurity_calc_func = None
        # 叶子节点值计算函数(如果是分类,可能是投票函数,如果是回归可能是均值函数或者取中值)
        self._leaf_value_calc_func = None

    def fit(self,X,y,loss=None):
        """
        根据输入数据构建决策树,并返回决策树根节点
        ---
        X: 输入样本数据特征集合
        y: 输入样本数据对应的标签集合
        loss: 输入可以选择的损失函数,gradient boosting方法需要
        ---
        """
        self.root = self._build_tree(X, y)
        self.loss = loss
        return self.root

    def _build_tree(self,X,y,curr_depth=0):
        """
        递归的构造决策树,并返回决策树的根节点
        参数:
        ---
        X: 输入样本集合的特征矩阵
        y: 输入样本集合的标签数据
        curr_depth: 在递归过程中标识当前的深度,用于判断是否进一步分割子树
        """

        # 首先判断输入的样本标签的维度,决定是否需要进行维度扩展

        if(len(np.shape(y))==1):
            # 如果输入的标签维度为1,说明需要扩展一下维度,为了方便与特征集pinjie
            y=np.expand_dims(y, axis=1)

        # 接下来为了方便将数据根据特征以及threshold进行分割,将标签y与特征集合X拼接起来
        Xy = np.concatenate((X,y), axis=1)
        
        num_samples = np.shape(X)[0]
        num_features = np.shape(X)[1]

        # 用于计算当前节点的可能的最大分割impurity值
        curr_largest_impurity = 0
        # 存储当前最佳分割点
        best_separate_point={}
        left_brunch_data = None
        right_brunch_data = None
        
        if num_samples >= self.minimum_sample and curr_depth<=self.max_depth:
            for feature_index in range(num_features):
                # 对每一个特征维度,进行impurity的计算
                # 然后对该特征的每一个可能的取值,作为阈值进行impurity计算
                unique_feature_values = np.lib.arraysetops.unique(X[:,feature_index])
                # 每一个可能的取值,都可能是最佳的threshold
                for threshold in unique_feature_values:
                    # 然后依据feature 以及 threshold进行数据的分割
                    Xy_left,Xy_right = separate_data_on_feature_by_threshold(Xy, feature_index, threshold)
                    if(len(Xy_left)==0 or len(Xy_right)==0):
                        continue
                    yleft = Xy_left[:,num_features:]
                    yright = Xy_right[:,num_features:]
                    # 然后调用impurity的计算函数计算当前这种分割的impurity值
                    curr_impurity = self._impurity_calc_func(y,yleft,yright)
                    if curr_impurity > curr_largest_impurity:
                        curr_largest_impurity = curr_impurity
                        best_separate_point={"feature_index":feature_index,"threshold":threshold}
                        left_brunch_data = {"X":Xy_left[:,:num_features],"y":Xy_left[:,num_features]}
                        right_brunch_data = {"X":Xy_right[:,:num_features],"y":Xy_right[:,num_features]}
        
        if curr_largest_impurity>self.minimum_impurity:
            left_brunch = self._build_tree(left_brunch_data["X"], left_brunch_data["y"],curr_depth+1)
            right_brunch = self._build_tree(right_brunch_data["X"], right_brunch_data["y"],curr_depth+1)
            return TreeNode(best_separate_point["feature_index"],
                best_separate_point["threshold"],None,left_brunch,right_brunch)
        # 到这里说明已经到叶子节点了,构造叶子节点
        leaf_value = self._leaf_value_calc_func(y)
        return TreeNode(value=leaf_value)


    def predict_value(self,x,root=None):
        """
        根据输入的样本特征x,预测样本的分类或者回归值
        参数:
        ---
        x:输入样本的特征向量
        root:可选的决策树根节点,不提供则默认使用当前树的root,作为根节点
        ---
        """
        if not root:
            root = self.root
        if root.value != None:
            return root.value
        feature_value = x[root.feature_index]
        braunch = root.left_brunch
        if isinstance(feature_value, (int,float)):
            if feature_value >= root.threshold:
                braunch = root.right_brunch
        elif feature_value == root.threshold:
            braunch = root.right_brunch
        # 递归判断子树
        return self.predict_value(x,braunch)


    def predict(self,X):
        """
        输入的样本集合,返回样本的分类或者回归集合(可以调用上面的predict_value方法对每一条样本进行处理)
        参数:
        ---
        X: 输入样本集合
        """
        if len(np.shape(X))==1:
            return [].append(self.predict_value(X))
        output=[]
        for x in X:
            output.append(self.predict_value(x))
        return output

    def print_tree(self,root=None,indent=" "):
        if root is None:
            root = self.root
        # 直接打印叶子节点的值
        if root.value is not None:
            print(root.value)
        else:
            print("{0}:{1}?".format(root.feature_index,root.threshold))
            print("{}left->".format(indent), end="")
            self.print_tree(root.left_brunch,indent+indent)
            print("{}right->".format(indent),end="")
            self.print_tree(root.right_brunch,indent+indent)


class ClassificationTree(DecisionTree):
    # 分类决策树 ,使用信息增益来度量impurity
    def _calc_information_gain(self,y,y_left,y_right):
        entropy_y = information_entropy(y)
        entropy_y_l = information_entropy(y_left)
        entropy_y_r = information_entropy(y_right)
        pl = len(y_left)/len(y)
        pr = len(y_right)/len(y)        
        return entropy_y-pl*entropy_y_l-pr*entropy_y_r
    # 使用投票法来选择节点标签
    def _calc_leaf_value_through_voting(self,y):
        unique_labels = np.lib.arraysetops.unique(y)
        maxLabel=None
        maxCount=0
        for label in unique_labels:
            lc = len(y[y==label])
            if lc >maxCount:
                maxCount = lc
                maxLabel = label
        return label
    def fit(self,X,y):
        self._impurity_calc_func = self._calc_information_gain
        self._leaf_value_calc_func = self._calc_leaf_value_through_voting
        return super(ClassificationTree, self).fit(X, y)

class ClassificationCARTTree(DecisionTree):
    # 分类决策树 ,使用信息增益来度量impurity
    def _calc_gini_scores(self,y,y_left,y_right):
        pl = len(y_left)/len(y)
        pr = len(y_right)/len(y)
        gini_all = calc_gini_group_score(y)        
        gini_left = calc_gini_group_score(y_left)
        gini_right = calc_gini_group_score(y_right)
        return gini_all-(pl*gini_left + pr*gini_right)
    # 使用投票法来选择节点标签
    def _calc_leaf_value_through_voting(self,y):
        unique_labels = np.lib.arraysetops.unique(y)
        maxLabel=None
        maxCount=0
        for label in unique_labels:
            lc = len(y[y==label])
            if lc >maxCount:
                maxCount = lc
                maxLabel = label
        return label
    def fit(self,X,y):
        self._impurity_calc_func = self._calc_gini_scores
        self._leaf_value_calc_func = self._calc_leaf_value_through_voting
        return super(ClassificationCARTTree, self).fit(X, y)

class RegressionTree(DecisionTree):
    # 回归树度量标准使用方差降低量
    def _calc_variance_reduction(self,y,y_left,y_right):
        variance_y = variance_cacl(y)
        variance_y_l = variance_cacl(y_left)
        variance_y_r = variance_cacl(y_right)
        pl = len(y_left)/len(y)
        pr = len(y_right)/len(y)
        return np.sum(variance_y-pl*variance_y_l - pr*variance_y_r)

    # 回归,叶子节点值计算方法,取均值
    def _cacl_mean_value(self,y):
        mean_val = np.mean(y,axis=0)
        # 如果标签是单变量则返回其值即可(因为np.mean返回的是list,取第一个即可)
        # 如果标签是多变量,则直接返回均值即可,也就是list
        return mean_val if len(mean_val)>1 else mean_val[0]
    def fit(self,X,y):
        self._impurity_calc_func = self._calc_variance_reduction
        self._leaf_value_calc_func = self._cacl_mean_value
        return super(RegressionTree,self).fit(X, y)


if __name__=="__main__":
    import pandas as pd
    data = pd.read_csv("../datasets/data_banknote_autentication.csv",delimiter=',',header=None)

    y=data.values[:,-1]
    X=data.values[:,:-1]

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.2,random_state=0)

    dtc = ClassificationTree()
    dtc.fit(train_x, train_y)

    pred = dtc.predict(test_x)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(test_y, pred))





"""
another implementation:
# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

# Load a CSV file
def load_csv(filename):
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
"""



