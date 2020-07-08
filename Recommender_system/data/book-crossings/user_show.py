"""挖掘Book-Crossings数据集中的信息"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class UserShow(object):
    def __init__(self):
        self.file_user = "bookcrossings/BX-Users.csv"
        self.file_book = "bookcrossings/Bx-Books.csv"
        self.file_ratings = "bookcrossings/Bx-Book-Ratings.csv"
        self.user_info = self.loadUserData()  # {userid: {"age":, "city", "state", "country"}}
        self.book_info = self.loadBookInfo()  # {isbn: book_name}
        self.user_book = self.loadUserRatings()  # {userid: isbn_list}

    def loadUserData(self):
        user_info = dict()
        for line in open(self.file_user, "r", encoding="ISO-8859-1"):
            if line.startswith("\"User-ID\""): continue
            if len(line.split("\";")) != 3: continue
            # 去除空格
            line = line.strip().replace(" ", "")
            # 去除字符串中的"
            userid, addr, age = [one.replace("\"", "") for one in line.split("\";")]
            if age=="NULL" or int(age) not in range(1, 120): continue
            # print(line)

            user_info.setdefault(userid, {})
            user_info[userid]["age"] = int(age)
            # Location分为三级，对应国、州、市
            if len(addr.split(",")) != 3: continue
            city, state, country = addr.split(",")
            user_info[userid]["city"] = city
            user_info[userid]["state"] = state
            user_info[userid]["country"] = country

        return user_info

    def loadBookInfo(self):
        """加载图书编号和名字的对应关系"""
        book_info = dict()
        for line in open(self.file_book, "r", encoding="ISO-8859-1"):
            if line.startswith("\"ISBN\""): continue
            isbn, book_name = line.replace("\"", "").split(";")[:2]
            book_info[isbn] = book_name
        return book_info

    def loadUserRatings(self):
        """获取每个用户评分大于5的图书信息"""
        user_book = dict()
        for line in open(self.file_ratings, "r", encoding="ISO-8859-1"):
            if line.startswith("\"User-ID\""): continue
            uid, isbn, rate = line.strip().replace("\"", "").split(";")
            user_book.setdefault(uid, list())
            if int(rate) > 5:
                user_book[uid].append(isbn)
        return user_book

    # 画条形图
    def plot_bar(self, X, Y, X_label, Y_label="数目"):
        plt.xlabel(X_label)
        plt.ylabel(Y_label)

        plt.xticks(np.arange(len(X)), X, rotation=90)  # 设置刻度
        for x, y in zip(np.arange(len(X)), Y):
            plt.text(x-0.2, y+3, str(y), rotation=30)
        plt.bar(np.arange(len(X)), Y)
        plt.show()

    def ageStatics(self):
        """统计不同年龄段的用户人数"""
        age_group_dict = dict()
        for key in self.user_info.keys():
            age = self.user_info[key]["age"]
            # 将年龄段处理成0-9=>0, 10-19=>1
            age_group = int(age) // 10
            age_group_dict.setdefault(age_group, 0)  # 类似get，如果存在该键，返回对应的键值
            age_group_dict[age_group] += 1

        age_group_sorted = sorted(age_group_dict.items(), key=lambda x: x[0], reverse=False)
        X = [x[0] for x in age_group_sorted]
        Y = [x[1] for x in age_group_sorted]
        self.plot_bar(X, Y, X_label="用户年龄段")

    def stateStatics(self):
        """统计美国不同州的用户人数"""
        state_group_dict = dict()
        for key in self.user_info.keys():
            if "state" in self.user_info[key].keys() and self.user_info[key]["state"] != "n/a" and \
                    self.user_info[key]["country"] == "usa":
                    state = self.user_info[key]["state"]
                    state_group_dict.setdefault(state, 0)
                    state_group_dict[state] += 1

        # print(state_group_dict)
        state_group_sorted = sorted(state_group_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(state_group_sorted)
        X = [x[0] for x in state_group_sorted]
        Y = [x[1] for x in state_group_sorted]
        self.plot_bar(X, Y, X_label="用户所处的州")


    def AgeBook(self):
        """统计不同年龄段喜欢的图书top10"""
        age_books = dict()
        age_books.setdefault(1, {})
        age_books.setdefault(2, {})

        for key in self.user_info.keys():
            if "country" not in self.user_info[key].keys(): continue
            if key not in self.user_book.keys(): continue

            age = int(self.user_info[key]["age"])
            if age in range(0, 30):
                for book in self.user_book[key]:
                    if book not in self.book_info.keys(): continue
                    age_books[1].setdefault(book, 0)
                    age_books[1][book] += 1

            if age in range(50, 120):
                for book in self.user_book[key]:
                    if book not in self.book_info.keys(): continue
                    age_books[2].setdefault(book, 0)
                    age_books[2][book] += 1

        first_group = age_books[1]
        print("0~30岁用户最喜欢的图书TOP 10：")
        top_ten_lists = sorted(first_group.items(), key=lambda x: x[1], reverse=True)[:10]
        print(top_ten_lists)
        same_books = []
        for item in top_ten_lists:
            print(self.book_info[item[0]])
            same_books.append(self.book_info[item[0]])

        second_group = age_books[2]
        print("50岁以上用户和30岁以下用户共同喜欢的图书：")
        top_ten_lists = sorted(second_group.items(), key=lambda x: x[1], reverse=True)[:10]
        for item in top_ten_lists:
            # print(self.book_info[item[0]])
            if self.book_info[item[0]] in same_books:
                print(self.book_info[item[0]])


user_show = UserShow()
#print(user_show.user_book.items())
#print(user_show.book_info)
user_show.stateStatics()
#user_show.AgeBook()