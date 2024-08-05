import copy
import torch
from torch import nn
import ray
from typing import Type
import torch.nn.functional as F

from parallel_linear.parallel_experts.moe import TaskMoE


# @ray.remote 
class MoEShell(torch.nn.Module):
    """Call a Sparsely gated mixture of experts layer with experts of type nn.Model.
    Args:
    num_tasks: needed number of gating layers
    num_experts: an integer - number of experts
    expert_type: the name of the type of experts, usually a subclass of torch.nn.Module
        ps. to make your model moe-able, make sure to implement functions as follows:
            filter_args(**kwargs): specifie the arguments needed in your class
            get_moe(filtered_args): build an instance of your class with the args, 
                                    through which each experts in MoE will be built.
    input/output_size: an integer - the i/o width of the moe layer
    e_out_size: an integer - the output size of each specific expert
    k: an integer - how many experts to use for each batch element
    cvloss=0: an float - 用于平衡系数变异(Coefficient of Variation, CV)损失; 
        CV损失用于正则化专家的选择, 鼓励均衡使用所有专家, 而不是集中使用少数几个。
    switchloss=0: an float - 用于平衡切换损失(Switch Loss); 切换损失,用于衡量专
        家选择的频率, 鼓励每个专家在批次中被均匀选择, 以防止特定专家过度使用或闲置。
    zloss=0: 系数, 用于平衡Z损失(Z Loss); Z损失用于防止专家门控网络输出的激活值过
        于集中, 促进输出的均匀分布, 增加模型的鲁棒性。
    bias=False: 布尔值, 指示是否在专家网络中使用偏置项; 如果设置为True, 专家网络中
        的线性层将包括偏置项; 如果为False, 则不包括偏置项。
    gating_activation=None: 门控网络的激活函数; 用于指定门控网络中使用的激活函数。
        如果未指定, 则默认使用ReLU。
    activation=None: 专家网络的激活函数; 用于指定专家网络中使用的激活函数。激活函数
        通常用于引入非线性, 提高模型的表达能力。
    noisy_gating=0: 浮点数, 指示是否在门控网络中使用噪声; 如果设置为0, 则在门控网络
        中不添加噪声, 否则添加以该数值为标准差的高斯噪声, 增加门控决策的随机性, 有助于均
        匀分配负载并防止过拟合。
    usage_mem=10000: 用于记录专家使用情况的内存大小; 用于统计专家选择频率, 辅助计算
        CV损失和切换损失。
    acc_aux_loss=False: 布尔值, 指示是否累积辅助损失; 如果设置为True, 则在训练过程
        中累积辅助损失, 这有助于在训练多个批次后进行更稳定的更新和正则化。
    noize_std: 门控网络噪声的标准差
    """
    def __init__(
            self, num_tasks:int, num_experts:int, original:torch.nn.Module,
            input_size, output_size, k, cvloss=0, switchloss=0, 
            zloss=0, bias=False, gating_activation=None, activation=None, 
            noisy_gating=0., usage_mem = 10000, acc_aux_loss=False, **kwargs):
        # 初始化父类对象
        super(MoEShell, self).__init__()
        # 填充基本信息
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.is_active = [True for _ in range(self.num_experts)]
        self.expert_score = [1.0 for _ in range(self.num_experts)]  # 各个专家的效率分,用于进行GPU资源的均衡调度
        self.noisy_gating = noisy_gating
        # self.input_size = input_size
        # self.output_size = output_size
        self.bias = bias
        self.k = min(k, self.num_experts)
        # 指定loss函数和激活函数的实现(这些loss通用吗?)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation
        self.acc_aux_loss = acc_aux_loss
        # 初始化与各loss有关的附加计数器
        if self.acc_aux_loss:
            self.init_aux_statistics()
        # 创建各个专家
        # self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        # self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.experts = []
        # expert_args = expert_type.filter_kwargs(**kwargs)
        for _ in range(num_experts):
            # new_expert = expert_type.get_moe(expert_args)
            new_expert = copy.deepcopy(original)
            self.experts.append(new_expert)
        self.experts = torch.nn.ModuleList(self.experts)
        # 创建门控网络(此处可以在sequencial中插入线性层和激活函数等处理)
        self.gatings = []
        for _ in range(num_tasks):
            new_gating = torch.nn.Sequential(
                torch.nn.Linear(input_size,
                2 * num_experts if noisy_gating else num_experts,
                bias=False)
            )
            self.gatings.append(new_gating)
        self.gatings = torch.nn.ModuleList(self.gatings)
        pass

    def forward(self, x:list):
        # TODO: 参考GPT代码实现: 
        # 1. forward没有task_id,不同任务一锅出
        # 2. 给专家也加上Remote装饰, 调用激活专家的forward方法时分配不同的GPU份额
        # 3. noise仅加在top-k的模型权重上,和加在全部专家分量上哪个更好?
        # 4. 保存每个专家的GPU分配评分,并按照训练用时动态调整,每次GPU份额分配参考该评分(归一化地)进行
        assert len(x) == self.num_tasks, 'moe block takes a list of input with lenth of self.num_tasks'
        # 分任务gating寻路结果
        gating_logits = []
        for task_idx in range(self.num_tasks):
            logits = self.gatings[task_idx](x[task_idx])
            if self.noisy_gating != 0.0:
                noise = torch.normal(0, self.noisy_gating, size=logits.size()).to(logits.device)
                logits = logits + noise
            gating_logits.append(logits)
        # top-k归一化, 计算:
        # 1. 每个任务的专家选取, 及各专家的份额
        # 2. 本轮需要进行推理的专家
        expert_selections = []
        active_experts = set([])
        for i in range(len(self.is_active)):
            self.is_active[i] = False
        for _ in range(self.num_tasks):
            topk_values, topk_indices = torch.topk(logits, self.k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
            topk_masked = mask * logits
            topk_normalized = topk_masked / (topk_masked.sum(dim=-1, keepdim=True) + 1e-9)
            expert_selections.append(topk_normalized)
            for i in range(len(topk_normalized)):
                if mask[i] == 1.0:
                    self.is_active[i] = True
                    active_experts.add(i)
        # Q: 各个任务一锅出后, 梯度更新是否会出现泄露;A:
        # 1. 没有激活的专家没有输出, 自然没有梯度更新
        # 2. 激活但没有被当前任务采用的专家, 其输出被当前任务的掩码层过滤掉了, 也不会产生梯度
        # 3. 只有激活且被采用的专家会发生梯度的积累和更新, 由于掩码层给出的权重不同,该梯度也相
        #   应不同,参数更新的幅度也相应地发生变化
        # 激活的专家进行推理
        e_outputs = {}
        for task_idx in range(self.num_tasks):
            for expert_idx in range(self.num_experts):
                if expert_selections[task_idx][expert_idx] == 0:
                    continue
                e_outputs[f't{task_idx}e{expert_idx}'] = self.experts[expert_idx](x[task_idx])
        # 根据所参考的专家和比率生成各个任务输出
        task_ouputs = []
        for task_idx in range(self.num_tasks):
            ans = None
            for e_idx in range(self.num_experts):
                if expert_selections[task_idx][e_idx] == 0:
                    continue
                if ans is None:
                    ans = e_outputs[e_idx] * expert_selections[task_idx][e_idx]
                else:
                    ans += e_outputs[e_idx] * expert_selections[task_idx][e_idx]
            assert ans is not None, f'task {task_idx} with selection {expert_selections[task_idx]} is getting a None in return'
            task_ouputs.append(ans)
        return task_ouputs

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        # self._gates = []
        # self._probs = []
        # self._logits = []
        # self._expert_sizes = []

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # loss = (self.cvloss * cvloss)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)

        # print("cvloss")
        # true_cvloss = self.compute_cvloss(torch.cat(self._gates, dim=0))
        # print(self.cvloss, cvloss, true_cvloss)

        # print("switchloss")
        # cat_probs = torch.cat(self._probs, dim=0)
        # true_switchloss = self.compute_switchloss(cat_probs, sum(self._expert_sizes))
        # print(self.switchloss, switchloss, true_switchloss)

        # print("zloss")
        # true_zloss = self.compute_zloss(torch.cat(self._logits, dim=0))
        # print(self.zloss, zloss, true_zloss)

        # assert torch.allclose(cvloss, true_cvloss)
        # assert torch.allclose(switchloss, true_switchloss)
        # assert torch.allclose(zloss, true_zloss)

        self.init_aux_statistics()
        return loss

    # def compute_topk_loss(self, probs):


    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss
    # # sample for filter_kwargs in models to be used in this moe structure
    # def filter_kwargs(**kwargs)->dict:
    #     remain_list = []
    #     ans = {k:v for k, v in kwargs.items() if k in remain_list}
    #     return ans
import torch
import torch.nn as nn
import torch.optim as optim

# # 定义一个简单的专家类
# class SimpleExpert(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(SimpleExpert, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
    
#     def forward(self, x):
#         return self.fc(x)
    
#     @staticmethod
#     def filter_kwargs(**kwargs):
#         remain_list = ['input_size', 'output_size']
#         ans = {}
#         for arg in remain_list:
#             if not arg in kwargs:
#                 assert False, f'Warning: arg {arg} is not found in kwargs'
#             ans[arg] = kwargs[arg]
#         return ans
    
#     @staticmethod
#     def get_moe(filtered_args):
#         return SimpleExpert(**filtered_args)

# # 定义一个简单的训练数据和标签
# input_size = 10
# output_size = 1
# batch_size = 5
# num_tasks = 2
# num_experts = 3
# k = 2

# # 创建MoE模型
# moe_model = MoEShell(
#     num_tasks=num_tasks, 
#     num_experts=num_experts, 
#     expert_type=SimpleExpert, 
#     input_size=input_size, 
#     output_size=output_size, 
#     k=k,
#     cvloss=0.1, 
#     switchloss=0.1, 
#     zloss=0.1
# )

# # 创建输入数据
# x = [torch.randn(batch_size, input_size) for _ in range(num_tasks)]
# y = [torch.randn(batch_size, output_size) for _ in range(num_tasks)]

# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(moe_model.parameters(), lr=0.01)

# # 训练模型
# for epoch in range(10):
#     optimizer.zero_grad()
    
#     # 前向传播
#     outputs = moe_model(x)
    
#     # 计算损失
#     losses = [criterion(outputs[i], y[i]) for i in range(num_tasks)]
#     total_loss = sum(losses)
    
#     # 反向传播
#     total_loss.backward()
    
#     # 更新参数
#     optimizer.step()
    
#     # 输出损失
#     print(f'Epoch [{epoch+1}/10], Loss: {total_loss.item():.4f}')
