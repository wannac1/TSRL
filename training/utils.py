# file: training/utils.py
# description: A standard AverageMeter utility class for tracking metrics.

class AverageMeter(object):
    """
    一个用于计算和存储平均值和当前值的通用工具类。
    常用于记录损失(loss)和准确率(accuracy)等指标。
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.clear() # 调用clear来初始化所有值为0

    def clear(self):
        """ 重置所有统计数据 """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新记录器的状态。
        Args:
            val (float): 当前批次的平均值 (例如, 当前batch的loss)。
            n (int): 当前批次的大小 (batch size)。
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        """ 返回全局平均值 """
        return self.avg