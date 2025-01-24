import torch.nn as nn
import torch
import torch.nn.functional as F
from Self_Attention import MultiHeadSelfAttention


class AttentionFusion(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num, output_size, dropout_rate=0):
        super(AttentionFusion, self).__init__()

        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_size) for _ in range(num)])

        self.num = num

        # Define the attention weights
        self.attention_weights = nn.Parameter(torch.rand(num))

        # Linear transformation for the fused representation
        self.fusion_linear = nn.Linear(hidden_size * num, hidden_size)
        # self.conv = MultiHeadSelfAttention(hidden_size * num, hidden_size, hidden_size, 10, 0)
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_list):

        for i, linear in enumerate(self.linear_layers):
            x_list[i] = F.relu(linear(x_list[i]))

        # Compute attention scores
        # attention_scores = F.softmax(self.attention_weights, dim=0)
        attention_scores = self.attention_weights

        # Apply attention to the inputs
        for i in range(self.num):
            x_list[i] = x_list[i] * attention_scores[i]

        # # 使用 torch.stack 将列表中的 tensors 堆叠起来
        # stacked_tensor = torch.stack(x_list)
        #
        # # 使用 torch.sum 将堆叠后的 tensors 相加
        # result_tensor = torch.sum(stacked_tensor, dim=0)

        result_tensor = torch.cat(x_list, dim=1)

        # Fuse the attention-weighted inputs
        fused_representation = F.relu(self.fusion_linear(result_tensor))
        # fused_representation = F.relu(self.conv(result_tensor))

        # Apply dropout before the final classification layer
        fused_representation = self.dropout(fused_representation)

        # Apply the final classification layer with softmax activation
        output = F.softmax(self.output_linear(fused_representation), dim=1)

        return output


class NNFusion(nn.Module):
    def __init__(self, input_size,
                 hidden_size, num, output_size, dropout_rate=0):
        super(NNFusion, self).__init__()

        self.num = num

        self.fusion_linear = nn.Linear(input_size * num, hidden_size)

        self.output_linear = nn.Linear(hidden_size, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_list):
        x = torch.cat([x for x in x_list], dim=1)

        fused_representation = F.relu(self.fusion_linear(x))

        fused_representation = self.dropout(fused_representation)

        output = F.softmax(self.output_linear(fused_representation), dim=1)

        return output
