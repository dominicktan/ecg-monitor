"""Model definitions for ECG beat classification."""

import torch
import torch.nn as nn


class HybridCNN(nn.Module):
    """1D-CNN on waveform concatenated with tabular features before classifier.

    Architecture:
        Waveform branch: Conv1d(32) -> Conv1d(64) -> Conv1d(128) -> AdaptiveAvgPool
        Tabular branch: Linear(64) -> BN -> ReLU -> Dropout
        Fusion: concatenate -> Linear(64) -> ReLU -> Linear(num_classes)
    """

    def __init__(self, seq_len, n_tabular, num_classes):
        super().__init__()
        self.wf_len = seq_len
        self.waveform_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(n_tabular, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        wf = x[:, :self.wf_len].unsqueeze(1)
        tab = x[:, self.wf_len:]
        wf_feat = self.waveform_branch(wf)
        tab_feat = self.tabular_branch(tab)
        combined = torch.cat([wf_feat, tab_feat], dim=1)
        return self.classifier(combined)
