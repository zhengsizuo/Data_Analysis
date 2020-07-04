"""
The course project in Pattern Recognition course.
Author: zhs
Date: Dec 12, 2018
"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

train_Set = pd.read_csv('data/train.csv')
test_Set = pd.read_csv('data/test.csv')
PassengerId = test_Set['PassengerId']
# all_data = pd.concat([train_Set, test_Set], ignore_index=True)

def analysis_relations():
    """观察各特征与存活与否的关系，从而决定采取的特征"""
    # Pclass与是否存活的图表关系
    gp = train_Set[['Pclass', 'Survived']].groupby(['Pclass'],
                                            as_index=False).mean().sort_values(by='Survived', ascending=False)
    print(gp)

    sns.barplot(x="SibSp", y="Survived", data=train_Set, palette='Set3')
    sns.barplot(x="Parch", y="Survived", data=train_Set, palette='Set3')

    # facet = sns.FacetGrid(train_Set, hue="Survived", aspect=2)
    # facet.map(sns.kdeplot, 'Age', shade=True)
    # facet.set(xlim=(0, train_Set['Age'].max()))
    # facet.add_legend()
    #
    # # Embarked取一定值的情况下，Pclass、Sex分别与Survived的关系
    # grid = sns.FacetGrid(train_Set, row='Embarked', size=2.2, aspect=1.6)
    # grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    # grid.add_legend()
    #
    # # Embarked取一定值的情况下，Fare、Sex分别与Survived的关系
    # grid = sns.FacetGrid(train_Set, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    # grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    # grid.add_legend()
    #
    # # 分析Fare和年龄特征该怎么分组,qcut函数指每一个取值范围内样本数相同
    # train_Set['FareBand'] = pd.qcut(train_Set['Fare'], 4)
    # train_Set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
    #                                                                                             ascending=True)
    # train_Set['AgeBand'] = pd.cut(train_Set['Age'], 5)
    # train_Set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
    #                                                                                           ascending=True)
    # train_set = train_Set.drop(['AgeBand', 'FareBand'], axis=1)


def anomaly_detection(all_data):
    """异常检测，把测试集中的异常数据进行惩罚性修改"""
    # all_data = pd.concat([train_Set, test_Set], ignore_index=True)
    all_data['Surname'] = all_data['Name'].apply(lambda x: x.split(',')[0].strip())  # lambda隐式函数
    Surname_Count = dict(all_data['Surname'].value_counts())
    print(Surname_Count)
    all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x: Surname_Count[x])
    Female_Child_Group = all_data.loc[
        (all_data['FamilyGroup'] >= 2) & ((all_data['Age'] <= 12) | (all_data['Sex'] == 'female'))]
    Male_Adult_Group = all_data.loc[
        (all_data['FamilyGroup'] >= 2) & (all_data['Age'] > 12) & (all_data['Sex'] == 'male')]
    Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
    Female_Child.columns = ['GroupCount']
    Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
    Male_Adult.columns = ['GroupCount']

    Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
    Dead_List = set(Female_Child_Group[Female_Child_Group.apply(lambda x: x == 0)].index)
    print(Dead_List)
    Male_Adult_List = Male_Adult_Group.groupby('Surname')['Survived'].mean()
    Survived_List = set(Male_Adult_List[Male_Adult_List.apply(lambda x: x == 1)].index)
    print(Survived_List)

    # 把测试集中异常遇难组样本改成成年男性，把异常幸存者样本改成幼年女性
    train = all_data.loc[all_data['Survived'].notnull()]
    test = all_data.loc[all_data['Survived'].isnull()]
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Sex'] = 'male'
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Age'] = 60
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)), 'Title'] = 'Mr'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Sex'] = 'female'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Age'] = 5
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)), 'Title'] = 'Miss'

    train_Set = train.drop(['FamilyGroup', 'Surname'], axis=1)
    test_Set = test.drop(['FamilyGroup', 'Surname'], axis=1)
    return train_Set, test_Set


def data_clean(train_Set):
    """对原始数据中的字符型数据和缺失项进行处理"""
    # 用0替代'male'，用1替代'female'
    train_Set.loc[train_Set['Sex'] == 'male', 'Sex'] = 0
    train_Set.loc[train_Set['Sex'] == 'female', 'Sex'] = 1
    # 用C给‘Embarked’的缺失项填补
    train_Set['Embarked'] = train_Set.Embarked.fillna('C')
    # [train_Set.Embarked.isnull()] = train_Set.Embarked.dropna().mode().values  # 填充为C？
    train_Set.loc[train_Set['Embarked'] == 'S', 'Embarked'] = 0
    train_Set.loc[train_Set['Embarked'] == 'C', 'Embarked'] = 1
    train_Set.loc[train_Set['Embarked'] == 'Q', 'Embarked'] = 2

    train_Set['Sex'] = train_Set['Sex'].astype(int)
    train_Set['Embarked'] = train_Set['Embarked'].astype(int)

    # 用0填补‘Cabin’的缺失项，用1填补‘Cabin’的非缺失项
    # train_Set['Cabin'] = train_Set.Cabin.fillna(0)
    train_Set.loc[train_Set['Cabin'].notnull(), ['Cabin']] = 1
    train_Set.loc[train_Set['Cabin'].isnull(), ['Cabin']] = 0

    return train_Set

def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

def predict_age(train_Set, test_Set):
    all_data = pd.concat([train_Set, test_Set], ignore_index=True)
    # 用随机森林回归模型对年龄进行预测
    age_df = all_data[['Age', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(all_data['Age'].notnull())]
    age_df_isnull = age_df.loc[(all_data['Age'].isnull())]
    X = age_df_notnull.values[:, 1:]
    Y = age_df_notnull.values[:, 0]
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X, Y)
    predict_Ages = RFR.predict(age_df_isnull.values[:, 1:])
    all_data.loc[all_data['Age'].isnull(), ['Age']] = predict_Ages

    # 增加同票号特征
    Ticket_Count = dict(all_data['Ticket'].value_counts())
    all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x: Ticket_Count[x])
    all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

    train_Set = all_data[:891]
    test_Set = all_data[891:]
    return train_Set, test_Set


def create_new_feature(train_Set, test_Set):
    """根据原特征提取新的特征"""
    combine_data = [train_Set, test_Set]
    # 从Name中提取Title特征
    for data in combine_data:
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)  # 正则表达式
    # print(pd.crosstab(train_Set['Title'], train_Set['Sex']))
    for data in combine_data:
        data['Title'] = data['Title'].replace(['Capt', 'Col', 'Dr', 'Major', 'Rev'], 'officer')
        data['Title'] = data['Title'].replace(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royal')
        data['Title'] = data['Title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
        data['Title'] = data['Title'].replace(['Mlle', 'Miss'], 'Miss')
        data['Title'] = data['Title'].replace('Mr', 'Mr')
        data['Title'] = data['Title'].replace(['Master', 'Johkheer'], 'Master')
    # train_Set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    title_mapping = {"officer": 0, "Royal": 1, "Mrs": 2, "Miss": 3, "Mr": 4, "Master": 5}
    for data in combine_data:
        data['Title'] = data['Title'].map(title_mapping)
        data['Title'] = data['Title'].fillna(0)

    # 从Parch和Sibsp中提取FamilySize特征
    for dataset in combine_data:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    train_Set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False)
    # 提取IsAlone特征并删除FamilySize、Parch、Sibsp
    for dataset in combine_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_Set = train_Set.drop(['FamilySize', 'Parch', 'SibSp', 'Name'], axis=1)
    test_Set = test_Set.drop(['FamilySize', 'Parch', 'SibSp', 'Name'], axis=1)

    return train_Set, test_Set


def group_by(train_Set, test_Set):
    """将Fare和Age做分组"""
    combine_data = [train_Set, test_Set]
    # 将Age分组
    for dataset in combine_data:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # 将Fare分组
    test_Set['Fare'].fillna(test_Set['Fare'].dropna().median(), inplace=True)
    for dataset in combine_data:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)


def classify(X_train, Y_train, X_test):
    # Random Forest
    random_forest = RandomForestClassifier(random_state=10, warm_start=True,
                                           n_estimators=26, max_depth=6, max_features='sqrt')
    random_forest.fit(X_train, Y_train)
    Y_predict = random_forest.predict(X_test)
    random_forest.score(X_train, Y_train)
    acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    print(acc_random_forest)

    # 支持向量机
    # linear_svc = SVC()
    # linear_svc.fit(X_train, Y_train)
    # Y_predict = linear_svc.predict(X_test)
    # acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    # print(acc_linear_svc)

    return Y_predict.astype(int)


if __name__ == '__main__':
    Sex_gp = train_Set[['Embarked', 'Survived']].groupby(['Embarked'],
                                                   as_index=False).mean().sort_values(by='Survived', ascending=False)
    # train_Set['FareBand'] = pd.qcut(train_Set['Fare'], 4)
    # FareBand_gp = train_Set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
    #                                                                                             ascending=True)
    # train_Set['AgeBand'] = pd.cut(train_Set['Age'], 5)
    # AgeBand_gp = train_Set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand',
    #                                                                                           ascending=True)
    # analysis_relations()
    # all_data = pd.concat([train_Set, test_Set], ignore_index=True)
    #
    # train_Set, test_Set = anomaly_detection(all_data)
    # train_Set = data_clean(train_Set)
    # test_Set = data_clean(test_Set)
    # train_Set, test_Set = predict_age(train_Set, test_Set)
    #
    # train_Set = train_Set.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
    # test_Set = test_Set.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
    # train_Set, test_Set = create_new_feature(train_Set, test_Set)
    # group_by(train_Set, test_Set)
    #
    # X_train = train_Set.drop('Survived', axis=1)
    # Y_train = train_Set['Survived']
    # X_test = test_Set.drop("Survived", axis=1).copy()
    #
    # y_predict = classify(X_train, Y_train, X_test)
    # submission = pd.DataFrame({
    #     "PassengerId": PassengerId,
    #     "Survived": y_predict
    # })
    # submission.to_csv('submission10.csv', index=False)


