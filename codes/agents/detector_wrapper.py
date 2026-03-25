# file: training/agents/detector_wrapper.py
# description: A wrapper to make any detector plug-and-play for the RL Tutor,
#              without modifying the original detector's source code.

import torch
import torch.nn as nn

class DetectorWrapper(nn.Module):

    def __init__(self, detector_model, feature_layer_name):

        super().__init__()
        self.detector = detector_model
        self.feature_layer_name = feature_layer_name
        
        self.features = None
        
        self._register_hook()

    def _hook_fn(self, module, input, output):

        self.features = output

    def _register_hook(self):

        found = False
        for name, layer in self.detector.named_modules():
            if name == self.feature_layer_name:
                layer.register_forward_hook(self._hook_fn)
                found = True
                break
        if not found:
            raise AttributeError(f" '{self.feature_layer_name}' ")
            

        return self.detector(x)

    def extract_features(self, x):

        _ = self.detector(x)
        

        features = self.features.clone()
        
        if features.dim() > 2:
            features = torch.flatten(features, start_dim=1)
            
        return features

    def get_feature_dim(self):

        dummy_input = torch.randn(1, 3, self.detector.config['img_size'], self.detector.config['img_size'])
        dummy_input = dummy_input.to(next(self.detector.parameters()).device)
        
        features = self.extract_features(dummy_input)
        return features.shape[1]