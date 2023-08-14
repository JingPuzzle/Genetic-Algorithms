import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class FGA_D01KP(object):
    def __init__(self, path1, path2, KP_C, iteration=500, person_num=30, ga_choose_ratio=0.2, cross_ratio=0.5,
                 mutate_ratio=0.1):
        self.KP_P = self.read_data(path1)  # 价值系数集
        self.KP_W = self.read_data(path2)  # 重量系数集
        self.KP_C = KP_C  # 背包载重
        self.KP_H, self.KP_P, self.KP_W = self.sort_p_w(self.KP_P, self.KP_W)  # 对数组进行排序并存储原下标至H
        self.KP_N = len(self.KP_P)  # 3n
        self.iteration = iteration  # 迭代次数
        self.person_num = person_num  # 个体数
        self.ga_choose_ratio = ga_choose_ratio  # 选择父母的比例
        self.cross_ratio = cross_ratio  # 交叉概率
        self.mutate_ratio = mutate_ratio  # 变异概率
        self.iter_best_fval = []  # 每次迭代的最优值
        self.best_person = []  # 最好的个体
        self.best_fval = 0  # 最好的值

    def read_data(self, path):  # 读取数据集
        data = pd.read_csv(path, sep=',', header=None)
        data = data.loc[:, :2]
        return data

    def sort_p_w(self, KP_P, KP_W):  # 对数组进行排序并存储原下标
        tmp_list = list()
        new_P = []
        new_W = []
        for i in range(len(KP_P)):
            for j in range(3):
                new_list = [KP_P.iloc[i, j], KP_W.iloc[i, j]]
                tmp_list.append(new_list)
                new_P.append(KP_P.iloc[i, j])
                new_W.append(KP_W.iloc[i, j])
        tmp_list = list(enumerate(tmp_list))
        tmp_list = sorted(tmp_list, key=lambda x: x[1][0] / x[1][1])
        new_H = [x[0] for x in tmp_list]
        return new_H, new_P, new_W

    def find_indices_of_element(self, lst, element):    # 寻找为element的列表位置
        indices = []
        for index, value in enumerate(lst):
            if value == element:
                indices.append(index)
        return indices

    def ga_stochastic_repair(self, onePerson):  # 采用随机策略对个体进行修复
        KP_Y = [0] * self.KP_N # 二进制向量Y
        KP_Flag = [0] * int(self.KP_N / 3) # 标记向量Flag
        fweight = 0
        fvalue = 0
        KP_indices1 = self.find_indices_of_element(onePerson, 1)   # 获取下标为1
        random.shuffle(KP_indices1)  # 打乱顺序
        for index in KP_indices1: # 采用随机策略进行修复
            if (fweight + self.KP_W[index] <= self.KP_C) and (KP_Flag[math.floor(index/3)] == 0):
                fweight = fweight + self.KP_W[index]
                KP_Y[index] = 1
                KP_Flag[math.floor(index / 3)] = 1
        for i in range(self.KP_N):  # 采用贪心策略进一步优化
            if (fweight + self.KP_W[self.KP_H[i]] <= self.KP_C) and (KP_Flag[math.floor(self.KP_H[i] / 3)] == 0):
                fweight = fweight + self.KP_W[self.KP_H[i]]
                KP_Y[self.KP_H[i]] = 1
                KP_Flag[math.floor(self.KP_H[i] / 3)] = 1
        for i in range(self.KP_N):
            fvalue += KP_Y[i] * self.KP_P[i]
        return KP_Y, fvalue


    def initialPopolation(self):  # 初始化种群
        total_population = []
        p = 0.5  # 设置生成0的概率，这个可以调整，以达到最优
        while len(total_population) != self.person_num:
            person = []
            for i in range(self.KP_N):
                if random.random() < p:
                    person.append(0)
                else:
                    person.append(1)
            total_population.append(person)
        return total_population

    def ga_parent(self, fval_list):  # 遗传父母
        sort_index = np.argsort(fval_list)  # 排序然后提取它排序前的位置
        sort_index = sort_index[0:int(self.ga_choose_ratio * len(sort_index))]  # 按照ratio选择父母的大小
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.population[index])
            parents_score.append(fval_list[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):  # 选择算子，轮盘赌
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return genes_choose[index1], genes_choose[index2]

    def ga_cross(self, x, y):  # 交叉算子
        len_ = len(x)
        assert len(x) == len(y)
        path_list = [t for t in range(len_)]
        order = list(random.sample(path_list, 2))
        order.sort()
        start, end = order
        # 不需要进行冲突处理
        # 交叉
        tmp = x[start:end].copy()
        x[start:end] = y[start:end]
        y[start:end] = tmp

        return x, y

    def ga_mutate(self, gene):  # 变异算子
        path_list = [t for t in range(len(gene))]
        order = list(random.sample(path_list, int(len(gene) * self.mutate_ratio)))
        for i in order:
            if gene[i] == 0:
                gene[i] = 1
            else:
                gene[i] = 0
        return gene

    def run(self):  # 开始迭代
        self.population = self.initialPopolation()  # 个体
        for i in range(1, self.iteration + 1):
            tmp_index = 0  # 对应的序号
            tmp_best_fval = 0  # 当前最好的
            tmp_fval_list = []  # 价值的列表
            for j in range(int(self.person_num)):  # 修复个体并求得最优值
                self.population[j], tmp_fval = self.ga_stochastic_repair(self.population[j])
                tmp_fval_list.append(tmp_fval)
                if tmp_best_fval < tmp_fval:
                    tmp_best_fval = tmp_fval
                    tmp_index = j
            if self.best_fval < tmp_fval:
                self.best_fval = tmp_best_fval
                self.best_person = self.population[tmp_index]
            self.iter_best_fval.append(self.best_fval)
            print("第", i, "代")
            print("maxFval=", self.best_fval)
            # 选择部分优秀个体作为父代候选集合
            parents, parents_fval = self.ga_parent(tmp_fval_list)
            # 新的种群population
            population = parents.copy()
            # 生成新的种群
            while len(population) < self.person_num:
                # 轮盘赌方式选择
                gene_x, gene_y = self.ga_choose(parents_fval, parents)
                # 交叉
                if np.random.rand() < self.cross_ratio:
                    gene_x, gene_y = self.ga_cross(gene_x, gene_y)
                # 变异
                gene_x = self.ga_mutate(gene_x)
                gene_y = self.ga_mutate(gene_y)
                if len(population) < self.person_num:
                    population.append(gene_x)
                if len(population) < self.person_num:
                    population.append(gene_y)
            self.population = population
        tmp_index = 0  # 对应的序号
        tmp_best_fval = 0  # 当前最好的
        tmp_fval_list = []  # 价值的列表
        for j in range(int(self.person_num)):  # 修复个体并求得最优值
            self.population[j], tmp_fval = self.ga_stochastic_repair(self.population[j])
            tmp_fval_list.append(tmp_fval)
            if tmp_best_fval < tmp_fval:
                tmp_best_fval = tmp_fval
                tmp_index = j
        if self.best_fval < tmp_fval:
            self.best_fval = tmp_best_fval
            self.best_person = self.population[tmp_index]
        print("第", self.iteration, "代")
        print("maxFval=", self.best_fval)
        return self.best_person, self.iter_best_fval


if __name__ == '__main__':
    model = FGA_D01KP('data/WDKP/WDKP5p.csv', 'data/WDKP/WDKP5w.csv', 326793, 100, 50, 0.2, 0.2, 0.01)
    best_person, iter_best_fval = model.run()
    print("最好的个体：", best_person)
    iterations = [i for i in range(len(iter_best_fval))]
    plt.plot(iterations, iter_best_fval)
    plt.title("Convergence curve")
    plt.savefig("img/FGA_D01KP_Model2.png")
    plt.show()
