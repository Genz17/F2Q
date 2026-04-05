import numpy
import itertools

class ExactUCBBandit:
    def __init__(self, qNum, localSize, alpha=0.8, c=2.0, initial_q=20.0):
        """
        精确组合 UCB 老虎机 (完美映射 C_N^k)
        
        :param qNum: 总比特数 (如 20)
        :param localSize: 局部块大小 (如 4)
        :param alpha: 遗忘率 (0.0~1.0)，应对系统被“榨干”的情况
        :param c: UCB 探索常数。设大一点鼓励全局扫荡，设小一点鼓励局部深挖。
        :param initial_q: 乐观初始化分。
        """
        # 1. 静态生成所有可能的唯一组合 (对于 C(20,4) 是 4845 种)
        self.combinations = list(itertools.combinations(range(qNum), localSize))
        self.num_arms = len(self.combinations)
        
        self.alpha = alpha
        self.c = c
        
        # 2. 状态记录
        self.q_values = numpy.ones(self.num_arms) * initial_q # 期望收益
        # 初始化为 1 而不是 0，巧妙避开公式中除以 0 的错误
        self.n_pulls = numpy.ones(self.num_arms)              
        self.total_steps = 1                                  
        
    def select_block(self):
        """计算 UCB 分数并选择最佳组合"""
        self.total_steps += 1
        
        # 核心：计算每个组合的 UCB 分数
        # numpy 向量化运算，瞬间算出几千个组合的分数
        exploration_term = self.c * numpy.sqrt(numpy.log(self.total_steps) / self.n_pulls)
        ucb_scores = self.q_values + exploration_term
        
        # 加入极微小的噪声，防止在分数完全相同时 (比如开局) 永远固定按顺序选
        noisy_ucb = ucb_scores + numpy.random.normal(0, 1e-6, self.num_arms)
        
        # 找出 UCB 分数最高的臂 (索引)
        best_arm_idx = numpy.argmax(noisy_ucb)
        
        # 返回被选中的索引，以及实际的量子比特组合 (转成 list 以匹配你的代码)
        return best_arm_idx, list(self.combinations[best_arm_idx])

    def update(self, arm_idx, reward):
        """更新被选中组合的状态"""
        self.n_pulls[arm_idx] += 1
        # 使用 EMA 更新 Q 值，榨干后迅速遗忘
        self.q_values[arm_idx] = (1 - self.alpha) * self.q_values[arm_idx] + self.alpha * reward
