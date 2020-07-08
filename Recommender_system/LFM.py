"""
Latent factor model
Author: zhs
Date: July 6, 2020
"""
import os
import torch
import random
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LATENT_NUM = 5  # latent factor的数目
THRESHOLD = 2  # 分割正负样本的阈值
LAMBDA = 0.01  # 正则化参数
LR = 0.01  # 学习率
EPOCHES = 5  # 迭代次数
FILE = 'data/ml-1m/ratings.dat'


class LFM:
    def __init__(self, file_path):
        self.train_df, self.test_df = self.load_data(file_path)
        self.P, self.Q = self.init_matrix()
        self.R_dict = self.rand_generate_neg_sample()

    def load_data(self, file_path):
        ratings = pd.read_table(file_path, header=None, sep="::", names=['userID', 'movieID', 'rate', 'timestamp'])
        train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=0)  # 固定random_state每次生成的都相同
        return train_df, test_df

    def init_matrix(self):
        user_dict = self.train_df.groupby('userID')['userID'].count()
        movie_dict = self.train_df.groupby('movieID')['movieID'].count()
        # print("user numbers: ", len(user_dict))  # 6040
        print("item numbers: ", len(movie_dict))  # 3976

        P = np.random.rand(len(user_dict), LATENT_NUM)
        Q = np.random.rand(len(movie_dict), LATENT_NUM)

        self.movie_id_hash = dict()  # 从movieID到index的哈希映射
        for i in range(len(movie_dict)):
            self.movie_id_hash[movie_dict.index[i]] = i
        init_data = np.zeros([len(user_dict), len(movie_dict)])
        self.R_df = pd.DataFrame(init_data, index=user_dict.keys(), columns=movie_dict.keys())  # 评分矩阵
        self.test_dict = dict()  # 测试集
        for item in self.test_df.values:
            user_id = item[0]
            movie_id = item[1]
            rate = item[2]
            self.test_dict.setdefault(user_id, {})
            if rate > THRESHOLD:
                self.test_dict[user_id][movie_id] = 1.0

        return P, Q

    def rand_generate_neg_sample(self):
        R_dict = dict()  # 包含正负样本的评分字典
        file_name = 'data/ml-1m/R_dict_' + str(THRESHOLD) + '.dict'
        if os.path.exists(file_name):
            print("加载正负样本字典......")
            R_dict = pickle.load(open(file_name, 'rb'))
        else:
            for item in self.train_df.values:
                user_id = item[0]
                movie_id = item[1]
                rate = item[2]
                self.R_df.at[user_id, movie_id] = rate

            for i in self.R_df.index:
                print("user id: ", i)
                R_dict.setdefault(i, {})
                user_i = self.R_df.loc[i, :]

                pos_item_df = user_i[user_i > THRESHOLD]  # 认为评分大于THRESHOLD的为正样本
                for pos in pos_item_df.index:
                    R_dict[i][pos] = 1.0

                neg_item_df = user_i[user_i <= THRESHOLD]  # 认为评分小于THRESHOLD的为负样本
                # 保证正负样本1比1
                if len(pos_item_df) < len(neg_item_df):
                    neg_item = random.sample(list(neg_item_df.index), len(pos_item_df))
                else:
                    neg_item = neg_item_df.index
                for neg in neg_item:
                    R_dict[i][neg] = 0.0

            pickle.dump(R_dict, open(file_name, 'wb'))

        return R_dict

    def lfm_train(self):
        file_name = 'data/ml-1m/lfm_' + str(THRESHOLD) + '.model'
        if os.path.exists(file_name):
            print("加载P和Q矩阵......")
            P_list, Q_list = pickle.load(open(file_name, 'rb'))
        else:
            P_list = []
            Q_list = []
            for p in self.P:
                P_list.append(torch.tensor(p, requires_grad=True))

            for q in self.Q:
                Q_list.append(torch.tensor(q, requires_grad=True))
            for e in range(EPOCHES):
                print("Epoch now: ", e)
                for u, item_dict in self.R_dict.items():
                    p_u = P_list[u - 1]
                    for i in item_dict.keys():
                        R_ui = self.R_dict[u][i]
                        idx = self.movie_id_hash[i]
                        q_i = Q_list[idx]

                        R_ui_hat = torch.matmul(p_u, q_i.T)
                        loss = (R_ui - R_ui_hat) ** 2 + LAMBDA*(torch.norm(p_u, p=2) + torch.norm(q_i, p=2))
                        print("Epoch: ", e, "\t UserID:", u, "\t Loss: ", loss.item())
                        loss.backward()
                        with torch.no_grad():
                            p_u -= LR * p_u.grad
                            q_i -= LR * q_i.grad

                            # Manually zero the gradients after updating weights
                            p_u.grad.zero_()
                            q_i.grad.zero_()

            pickle.dump((P_list, Q_list), open(file_name, 'wb'))

        return P_list, Q_list

    def recommend(self, P_list, Q_list, user_id, top_n=40):
        rank_dict = dict()
        p_u = P_list[user_id-1].detach().numpy()
        train_items = set(self.train_df[(self.train_df.userID==user_id) & (self.train_df.rate>THRESHOLD)]['movieID'])
        # train_items = R_df.loc[user_id, :]
        # train_items = train_items[train_items > THRESHOLD].index
        for movie_id in self.R_df.columns:
            if movie_id in train_items:
                continue

            movie_idx = self.movie_id_hash[movie_id]
            q_i = Q_list[movie_idx].detach().numpy()
            r_ui = np.dot(p_u, q_i.T)
            rank_dict[movie_id] = r_ui

        return dict(sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def evaluate(self, P_list, Q_list, top_n=40):
        print("Evaluation start ...")
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for user in self.R_df.index:
            print("Evaluating the user ID: ", user)
            test_movies = self.test_dict.get(user, {})
            rec_movies = self.recommend(P_list, Q_list, user, top_n=top_n)
            for movie, rate in rec_movies.items():
                if int(movie) in test_movies.keys():
                    hit += 1
                    # print("Hit: ", hit)
                all_rec_movies.add(movie)
            rec_count += top_n
            test_count += len(test_movies)

        precision = hit / rec_count  # 所推荐的电影有多少是测试集中本来有的
        recall = hit / test_count  # 测试集中有多少电影进入了推荐列表
        coverage = len(all_rec_movies) / (1.0 * len(self.R_df.columns))
        print('precision=%.4f \t recall=%.4f \t coverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    lfm = LFM(FILE)
    P_list, Q_list = lfm.lfm_train()
    lfm.evaluate(P_list, Q_list)


