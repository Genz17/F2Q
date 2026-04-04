import numpy

class NonStationaryCombinatorialBandit:
    def __init__(self, num_qubits, num_select, epsilon=0.2, alpha=0.4, initial_q=20.0):
        """
        带有遗忘机制的组合多臂老虎机。
        
        :param num_qubits: 总量子比特数 (例如你的 20)
        :param num_select: 每次选取的局部块大小 (例如 4)
        :param epsilon: 探索率 (0.0~1.0)。例如 0.2 表示 20% 的时间完全随机选取，80% 的时间贪心选取。
        :param alpha: 学习率/遗忘率 (0.0~1.0)。值越大，遗忘历史越快。
                      因为一个块被优化后立刻就没有潜力了，所以 alpha 建议设大一点（如 0.4 - 0.6）。
        :param initial_q: 乐观初始化分数。设为一个较大的正数（如初期预期的最大 weight 下降值），
                          可以强制算法在初期“遍历探索”所有可能的量子比特。
        """
        self.num_qubits = num_qubits
        self.num_select = num_select
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_q = initial_q
        
        # 为每个量子比特维护一个期望收益 (Q-value)
        self.q_values = numpy.ones(num_qubits) * initial_q
        
    def select_block(self):
        """选择一组量子比特"""
        if numpy.random.rand() < self.epsilon:
            # 探索 (Exploration)：完全随机选取
            return numpy.random.choice(self.num_qubits, self.num_select, replace=False)
        else:
            # 利用 (Exploitation)：选取 Q-value 最高的 num_select 个 qubits
            # 加入微小的随机噪声，防止在分数相同时永远固定选取索引靠前的那几个
            noisy_q = self.q_values + numpy.random.normal(0, 1e-5, self.num_qubits)
            # argsort 会从小到大排序，取最后 num_select 个即为最大的
            best_indices = numpy.argsort(noisy_q)[-self.num_select:]
            # 打乱一下顺序返回
            numpy.random.shuffle(best_indices)
            return best_indices

    def update(self, selected_qubits, reward):
        """
        根据局部优化的结果更新被选中量子比特的分数。
        使用 EMA (指数移动平均) 实现自然遗忘：Q = (1 - alpha) * Q + alpha * Reward
        """
        for q in selected_qubits:
            self.q_values[q] = (1 - self.alpha) * self.q_values[q] + self.alpha * reward
        for q in selected_qubits:
            # 如果 reward = 0 (停滞期)，Q 值会自动按照 (1-alpha) 的比例衰减
            self.q_values[q] += reward
        max_q = numpy.max(self.q_values)
        if max_q < 0.1:
            self.q_values = self.q_values / max_q * 10
