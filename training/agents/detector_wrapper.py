# file: training/agents/detector_wrapper.py
# description: A wrapper to make any detector plug-and-play for the RL Tutor,
#              without modifying the original detector's source code.

import torch
import torch.nn as nn

class DetectorWrapper(nn.Module):
    """
    Detector包装器:
    将一个任意的detector模型包装起来，使其能够与RL_Trainer无缝协作。
    它通过注册一个前向钩子来获取中间层特征，从而实现了“即插即用”。
    """
    def __init__(self, detector_model, feature_layer_name):
        """
        Args:
            detector_model (nn.Module): 原始的、未经修改的detector模型实例。
            feature_layer_name (str): 我们希望从中提取特征的层的名称。
                                      例如 'avgpool', 'fc1', 'layer4'等。
        """
        super().__init__()
        self.detector = detector_model
        self.feature_layer_name = feature_layer_name
        
        # self.features是一个用于临时存储特征的变量
        self.features = None
        
        # 寻找并注册钩子
        self._register_hook()

    def _hook_fn(self, module, input, output):
        """
        这是钩子函数。每次模型前向传播到我们指定的层时，
        这个函数就会被自动调用，它的output参数就是该层的输出，即我们想要的特征。
        """
        self.features = output

    def _register_hook(self):
        """
        在detector的子模块中找到我们指定的层，并把_hook_fn注册到它上面。
        """
        found = False
        for name, layer in self.detector.named_modules():
            if name == self.feature_layer_name:
                # 注册一个前向钩子。handle可以用来移除钩子，但这里我们不需要。
                layer.register_forward_hook(self._hook_fn)
                found = True
                break
        if not found:
            raise AttributeError(f"指定的特征层 '{self.feature_layer_name}' 在模型中未找到。")
            
    def forward(self, x):
        """
        执行完整的分类任务，行为与原始detector完全一致。
        """
        return self.detector(x)

    def extract_features(self, x):
        """
        提取我们感兴趣的中间层特征。
        """
        # 为了触发钩子，我们仍然需要执行一次完整的前向传播
        # 但我们不关心它的最终输出，因为钩子函数会自动把需要的特征存入self.features
        _ = self.detector(x)
        
        # 钩子函数执行后，self.features就已经被填充了
        # 注意需要clone()以防后续计算影响到计算图
        features = self.features.clone()
        
        # 对于池化层等输出，可能需要展平
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
            
        return features

    def get_feature_dim(self):
        """
        动态推断特征维度。
        """
        # 创建一个假的输入张量来执行一次前向传播
        dummy_input = torch.randn(1, 3, self.detector.config['img_size'], self.detector.config['img_size'])
        dummy_input = dummy_input.to(next(self.detector.parameters()).device)
        
        # 提取特征并返回其维度
        features = self.extract_features(dummy_input)
        return features.shape[1]