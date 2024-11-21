import torch.nn as nn
import torch
import torch.nn.functional as F
class Expert(nn.Module):
    def __init__(self, input_dim, target_dim):
        super(Expert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2 + target_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2 + target_dim // 2, target_dim)
        )
    def forward(self, x):
        return self.expert(x)
    
    
class TopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        super(TopKGating, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gating_scores = self.gate(x)
        top_k_values, top_k_indices = torch.topk(F.softmax(gating_scores, dim=1), self.top_k)
        return top_k_indices, top_k_values


class MoE(nn.Module):
    def __init__(self, input_dim, target_dim, num_experts, top_k):
        super(MoE, self).__init__()
        self.experts = []
        self.num_experts = num_experts
        
        self.gating = nn.Linear(input_dim, 1)
        self.experts = nn.ModuleList([Expert(input_dim, target_dim) for _ in range(num_experts)])

        self.target_dim = target_dim
        self.top_k = top_k
        
    def forward(self, x):
        batch_size = x[0].shape[0]
        indices = []
        for i in range(self.num_experts):
            indices.append(self.gating(x[i]))
        indices = torch.cat(indices, axis=1)
        top_k_values, top_k_indices = torch.topk(F.softmax(indices, dim=1), self.top_k)
        
        expert_outputs = torch.zeros(batch_size, self.top_k, self.target_dim).to(x[0].device)
        x_expanded = []
        for i in range(self.num_experts):
            x_expanded.append(x[i].unsqueeze(1).expand(-1, self.top_k, -1))  # [batch_size, top_k, input_dim]
        expert_outputs = torch.zeros(batch_size, self.top_k, self.target_dim).to(x[0].device)

        for i in range(self.num_experts):
            mask = (top_k_indices == i).float().unsqueeze(-1)  # [batch_size, top_k, 1]
            selected_inputs = x_expanded[i] * mask  # [batch_size, top_k, input_dim]
            expert_outputs += self.experts[i](selected_inputs.view(-1, x[i].shape[1])).view(batch_size, self.top_k, self.target_dim) * mask

        gates_expanded = top_k_values.unsqueeze(-1).expand(-1, -1, self.target_dim)  # [batch_size, top_k, target_classes]

        x = (gates_expanded * expert_outputs).sum(1)
        
        
        return x, top_k_values 
