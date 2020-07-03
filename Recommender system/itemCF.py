"""
Item collaborating filter algorithm
Author: zhs
Date: July 1, 2020
"""
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class ItemCF:
    def __init__(self, file_path):
        self.train_df, self.test_dict = self.load_data(file_path)
        self.train_df = self.train_df.drop('userID', axis=1)
        print(self.train_df.index, self.train_df.columns)
        print("Test dictionary: ", self.test_dict)
        self.item_sim = self.itemSim()

    def load_data(self, file_path):
        np.random.seed(0)
        if os.path.exists('data/ml-1m/train_df.csv') and os.path.exists('data/ml-1m/test_df.csv'):
            print("加载训练集和测试集...")
            test_dict = dict()
            test_df = pd.read_csv('data/ml-1m/test_df.csv', index_col=None)
            for item in test_df.values:
                user_id = item[0]
                movie_id = item[1]
                rate = item[2]
                test_dict.setdefault(user_id, {})
                test_dict[user_id][movie_id] = rate
            return pd.read_csv('data/ml-1m/train_df.csv'), test_dict
        else:
            ratings = pd.read_table(file_path, header=None, sep="::", names=['userID', 'movieID', 'rate', 'timestamp'])
            train_df, test_df = train_test_split(ratings, test_size=0.2)

            user_dict = train_df.groupby('userID')['userID'].count()
            movie_dict = train_df.groupby('movieID')['movieID'].count()
            # print("userID number: ", user_dict.keys())
            # print("movieID number: ", movie_dict.keys())

            init_data = np.zeros([len(user_dict), len(movie_dict)])
            init_df = pd.DataFrame(init_data, index=user_dict.keys(), columns=movie_dict.keys())  # 用于构建同现矩阵的DataFrame
            test_dict = {}

            for item in train_df.values:
                user_id = item[0]
                movie_id = item[1]
                rate = item[2]
                init_df.at[user_id, movie_id] = rate

            for item in test_df.values:
                user_id = item[0]
                movie_id = item[1]
                rate = item[2]
                test_dict.setdefault(user_id, {})
                test_dict[user_id][movie_id] = rate

            init_df.to_csv('data/ml-1m/train_df.csv')
            test_df.to_csv('data/ml-1m/test_df.csv', index=None)

            return init_df, test_df

    def readComatrix(self):
        if os.path.exists('data/ml-1m/co_matrix2.csv'):
            print("加载同现矩阵...")
            return pd.read_csv('data/ml-1m/co_matrix2.csv')
        else:
            df = self.train_df.copy()
            co_occurence = np.zeros([len(df.columns), len(df.columns)])
            print(np.shape(co_occurence))
            co_df = pd.DataFrame(co_occurence, index=df.columns, columns=df.columns)

            for k in df.index:
                if k == 0:
                    continue
                print("user: ", k)
                temp_df = df.loc[k, :]
                temp_df = temp_df[temp_df > 0]
                for i in range(len(temp_df.index)-1):
                    for j in range(i+1, len(temp_df.index)):
                        id_i = temp_df.index[i]
                        id_j = temp_df.index[j]
                        co_df.at[id_i, id_j] += 1
                        co_df.at[id_j, id_i] += 1

            print(co_df)
            co_df.to_csv('data/ml-1m/co_matrix2.csv')
            return co_df

    def itemSim(self):
        if os.path.exists('data/ml-1m/item_sim2.json'):
            print("物品相似度从文件加载...")
            return json.load(open('data/ml-1m/item_sim2.json', 'r'))
        else:
            df = self.train_df.copy()
            user_num = len(df.index)
            N = df.apply(lambda x: user_num - x.value_counts().get(0, 0), axis=0)
            print("N(i):", N.index)
            print(N.values)

            co_matrix = self.readComatrix()
            # print(co_matrix.columns)
            co_matrix = co_matrix.reindex(index=co_matrix['movieID'])

            co_matrix = co_matrix.drop('movieID', axis=1)
            co_matrix.columns.name = 'movieID'
            co_matrix.index.name = 'movieID'
            print(co_matrix.index)
            print(co_matrix.columns)
            print("co-occurence matrix: ", co_matrix.head())
            item_sim = dict()

            for i in co_matrix.columns:
                print("Movie ID: ", i)
                item_sim.setdefault(i, {})
                temp_df = co_matrix.loc[:, i]  # 加载i列
                temp_df = temp_df[temp_df > 0]
                for j in temp_df.index:
                    cuv = temp_df[j]
                    if N.index[j] == '792':
                        print(cuv)
                    j = N.index[j]
                    # print(N[j])  # j是int64类型的
                    similarity = cuv / np.sqrt(N[i] * N[j])
                    item_sim[i].setdefault(j, 0)
                    item_sim[i][j] = similarity

            json.dump(item_sim, open('data/ml-1m/item_sim2.json', 'w'))
            return item_sim

    def recommend(self, user, k=8, nitems=40):
        """
        :param user: 用户
        :param k: k个临近物品
        :param nitems: 返回40个推荐
        :return:
        """
        result = dict()
        user = int(user) - 1
        u_items = self.train_df.loc[user, :]
        u_items = u_items[u_items > 0]
        # print(u_items)
        for i in u_items.index:
            # print("用户有过记录的物品：", i)
            for j, wij in sorted(self.item_sim[i].items(), key=lambda x: x[1], reverse=True)[:k]:
                if j in u_items.index:
                    # 如果以前有过评分记录，跳过
                    continue
                result.setdefault(j, 0)

                result[j] += u_items[i] * wij

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:nitems])

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self, k=8, n_items=40):
        print("Evaluation start ...")
        N = n_items
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for user in self.train_df.index:
            user = user + 1
            print("Evaluating the user ID: ", user)
            test_movies = self.test_dict.get(user, {})
            rec_movies = self.recommend(user)
            for movie, rate in rec_movies.items():
                if int(movie) in test_movies.keys():
                    hit += 1
                    print("Hit: ", hit)
                all_rec_movies.add(movie)
            rec_count += N
            test_count += len(test_movies)

        precision = hit / rec_count  # 所推荐的电影有多少是测试集中本来有的
        recall = hit / test_count  # 测试集中有多少电影进入了推荐列表
        coverage = len(all_rec_movies) / (1.0 * len(self.train_df.columns))
        print('precision=%.4f \t recall=%.4f \t coverage=%.4f' % (precision, recall, coverage))


FILE = 'data/ml-1m/ratings.dat'
item_cf = ItemCF(FILE)
# item_cf.readComatrix()
print(item_cf.recommend('1'))
item_cf.evaluate()