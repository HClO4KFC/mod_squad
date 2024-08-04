from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import os 
import numpy as np

from models_moe import PatchEmbed, MoEnhanceBlock, MoEnhanceTaskBlock

from timm.models.vision_transformer import VisionTransformer
from timm.models.mobilevit import mobilevit_xxs
from timm.models import ByobNet
from mbvit import MobileViT
# class MTMobileViT(VisionTransformer):
class MTMobileViT(MobileViT):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_types, embed_dim=768, global_pool=True, **kwargs):
        # debugging, should be disabled when switched to mbvit
        if False:
            kwargs.pop('pretrained')
            kwargs.pop('mbvit_version')
        super(MTMobileViT, self).__init__(embed_dim=embed_dim, **kwargs)
        self.taskGating = False
        self.ismoe = False
        # 多个mobilevit结合 TODO
        self.moe_type = 'normal'

        self.img_types = [type_ for type_ in img_types if type_ != 'rgb']
        assert global_pool == True
        del self.head
        norm_layer = kwargs['norm_layer']
        self.fc_norm = norm_layer(embed_dim)

        # create task head
        self.task_heads = []
        type_to_channel = {'depth_euclidean':1, 'depth_zbuffer':1, 'edge_occlusion':1, 'edge_texture':1, 'keypoints2d':1, 'keypoints3d':1, 'normal':3, 'principal_curvature':2,  'reshading':3, 'rgb':3, 'segment_semantic':18, 'segment_unsup2d':1, 'segment_unsup25d':1}
        # image_height, image_width = self.patch_embed.img_size
        # patch_height, patch_width = self.patch_embed.patch_size
        # assert image_height == 224 and image_width == 224
        for img_type in self.img_types:
            if 'class' in img_type:
                class_num = 1000 if img_type == 'class_object' else 365
                self.task_heads.append(
                    nn.Sequential(
                        nn.LayerNorm(self.embed_dim),
                        nn.Linear(self.embed_dim, class_num)
                    )
                )
            else:
                channel = self.type_to_channel[img_type]
                self.task_heads.append(
                    nn.Sequential(
                        nn.Conv2d(self.embed_dim, channel, kernel_size=1),
                        nn.Upsample(size=(self.image_height, self.image_width), mode='bilinear', align_corners=False)
                    )
                )
        self.task_heads = nn.ModuleList(self.task_heads)
        del self.head # 布置了多任务头, 原先的单任务头可以丢弃

        # self.task_embedding = nn.Parameter(torch.randn(1, len(self.img_types), embed_dim))

        if kwargs['init_weights']:
            self.apply(self._init_weights)

    def forward_features(self, x, task_rank, task):
        # B = x.shape[0]  # the size of this batch
        # x = self.patch_embed(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed

        # x = self.pos_drop(x)

        # # apply Transformer blocks

        # for blk in self.blocks:
        #     x = x + self.task_embedding[:,task_rank:task_rank+1, :]
        #     x = blk(x)

        # if 'class' in task:
        #     x = x[:, 1:, :].mean(dim=1)
        #     x = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     x = x[:, 1:, :]

        return x, 0

    def forward(self, x, task, get_flop=False):
        task_rank = -1
        for t, the_type in enumerate(self.img_types):
            if the_type == task:
                task_rank = t
                break
        assert task_rank > -1, f'错误: 没有找到task {task} 对应的任务编号.'

        x, z_loss = self.forward_features(x, task_rank, task)
        x = self.task_heads[task_rank](x)

        if get_flop:
            return x
        return x, z_loss

# from models_vit import VisionTransformer

def move_dict(ckpt, src, tgt):
    if src in ckpt and (src!=tgt):
        ckpt[tgt] = ckpt[src]
        del ckpt[src]

# A gating for a task
class MTMobileViTMoETaskGating(MTMobileViT):
    def __init__(self, img_types, num_heads=12, mlp_ratio=4., qkv_bias=True, # depth = 12,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None, att_w_topk_loss=0.0, att_limit_k=0, 
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True, moe_type='normal', 
                 switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss= 0.0, limit_k=0, 
                 w_MI = 0., noisy_gating=True, post_layer_norm=False, **kwargs):
        # w_topk_loss：表示Top-K损失的权重，用于在训练过程中对专家选择进行正则化。
        # 作用：通过增加一个额外的损失项，鼓励模型在选择专家时更均衡地使用不同的专家，防止某些专家被过度使用或闲置。
        # taskGating：表示是否启用任务门控机制。
        # 作用：任务门控机制允许模型根据任务动态选择合适的专家，从而提高多任务学习的效果。
        # ismoe：表示模型是否为MoE（混合专家）模型。
        # 作用：这个布尔值指示模型在前向传播时是否使用混合专家机制。
        # task_num：表示任务的数量。
        # 作用：用于确定有多少个不同的任务，每个任务可能有不同的头部和特征处理方式。
        # self.R：包含模型超参数的字典。
        # 作用：用于存储和传递一组关键的超参数，便于在模型的各个部分中引用和使用。
        # self.R 字典中的具体超参数
        # depth：Transformer块的数量。
        # 作用：控制模型的深度，影响模型的容量和特征提取的层次性。
        # task_num：任务的数量。
        # 作用：用于确定有多少个不同的任务。
        # head_dim：每个注意力头的维度。
        # 作用：影响多头注意力机制中每个头处理的特征维度。
        # noisy_gating：是否启用噪声门控。
        # 作用：通过在门控网络中加入噪声，提高专家选择的多样性和鲁棒性。
        # ffd_heads 和 ffd_noise：前馈网络中的头数和噪声。
        # 作用：控制前馈网络的结构和正则化，类似于多头注意力机制。
        # dim：嵌入维度。
        # 作用：表示输入特征的维度，影响模型的表示能力和计算复杂度。
        # num_heads：多头注意力机制中的头数。
        # 作用：通过增加注意力头数，提高模型捕捉不同特征的能力。
        # mlp_ratio：多层感知机（MLP）中隐藏层维度与输入维度的比率。
        # 作用：控制MLP中隐藏层的大小，影响模型的容量和计算复杂度。
        # qkv_bias：是否在Q、K、V线性层中使用偏置。
        # 作用：影响注意力机制的计算，通常启用以提高模型的表现力。
        # drop 和 attn_drop：Dropout的概率和注意力机制中的Dropout概率。
        # 作用：通过随机丢弃一部分神经元，防止过拟合，提高模型的泛化能力。
        # drop_path_rate：DropPath的概率。
        # 作用：增加残差路径上的随机性，进一步正则化模型。
        # norm_layer：归一化层的类型。
        # 作用：控制在模型中使用哪种归一化层，如LayerNorm或BatchNorm。
        # moe_type: MoE机制的类型。
        # 作用：指定使用哪种类型的混合专家机制，可能有不同的实现方式。
        # switchloss 和 zloss：用于MoE机制的正则化损失。
        # 作用：帮助均衡专家的使用频率，防止某些专家过度使用。
        # limit_k：选择的专家数量限制。
        # 作用：限制每次选择的专家数量，控制计算复杂度。
        # post_layer_norm：是否在每个层之后进行归一化。
        # 作用：提高训练的稳定性和模型的表现力。
        super(MTMobileViTMoETaskGating, self).__init__(
            img_types, num_heads=num_heads, mlp_ratio=mlp_ratio, # depth=depth, 
            qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)

        self.moe_type = moe_type
        # self.depth = depth

        self.w_topk_loss = w_topk_loss
        self.taskGating = True
        self.ismoe = True
        self.task_num = len(self.img_types)
        self.R = {
                'task_num': self.task_num, 'head_dim': head_dim, 'noisy_gating': noisy_gating,
                'ffd_heads': ffd_heads, 'ffd_noise': ffd_noise, 'num_heads': num_heads, 
                'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias, 'drop': drop_rate, 
                'attn_drop': attn_drop_rate, 'drop_path_rate': drop_path_rate, 
                'norm_layer': norm_layer, 'moe_type': moe_type, 'switchloss': switchloss, 
                'zloss': zloss, 'w_topk_loss': w_topk_loss, 'limit_k': limit_k,
                'post_layer_norm': post_layer_norm, # 'depth': depth
                }

        if self.is_moe:
            moe_target = {'BottleneckBlock'}
            
            for stage_idx in range(len(self.stages)):
                stage = self.stages[stage_idx]
                for expert_base_idx in range(len(stage)):
                    expert_base = stage[expert_base_idx]
                    experts = moelize(expert_base)
                



        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # self.twice_mlp = twice_mlp

        self.blocks = nn.Sequential(*[
            MoEnhanceTaskBlock(
                task_num=self.task_num,
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                moe_type=moe_type,switchloss=switchloss, zloss=zloss, w_topk_loss=w_topk_loss, limit_k=limit_k, 
                w_MI = w_MI, noisy_gating=noisy_gating, att_w_topk_loss=att_w_topk_loss,
                att_limit_k=att_limit_k, post_layer_norm=post_layer_norm,)
            for i in range(depth)])

        self.apply(self._init_weights)

    # reload 
    def pruning(self, args):
        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            vis_file = '/gpfs/u/home/AICD/AICDzich/scratch/' + str(args.copy) + '_vis.t7'
            load_file = '/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'
        else:
            vis_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/' + str(args.copy) + '_vis.t7'
            load_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'

        the_list = torch.load(vis_file)
        # print(the_list)
        all_experts = []
        
        dpr = [x.item() for x in torch.linspace(0, self.R['drop_path_rate'], self.R['depth'])]

        the_blocks = []
        # pruning_attn, pruning_mlp = False, False
        pruning_attn = [False] * self.depth
        pruning_mlp = [False] * self.depth
        for depth, blk in enumerate(self.blocks):
            expert_usage = the_list[depth][args.the_task] # a list of int for experts
            # mlp_bh = 1 if blk.attn.num_experts > blk.attn.num_heads else 0
            mlp_bh = 0

            num_attn_experts = blk.attn.num_heads
            if hasattr(blk.attn, 'num_experts'):
                if blk.attn.num_experts > blk.attn.num_heads:
                    mlp_bh = 1
                    choose = (np.array(expert_usage[0]) > args.thresh / blk.attn.num_heads)
                    num_attn_experts = int(choose.sum())

                    if num_attn_experts < blk.attn.num_heads: # threshold too large
                        ind = np.argpartition(np.array(expert_usage[0]), -blk.attn.num_heads)[-blk.attn.num_heads:]
                        choose[ind] = True
                        num_attn_experts = blk.attn.num_heads
                    pruning_attn[depth] = True
                else:
                    num_attn_experts = blk.attn.num_experts

            num_ffd_experts = 1
            if hasattr(blk.mlp, 'num_experts'):
                if blk.mlp.num_experts > blk.mlp.k:
                    choose = (np.array(expert_usage[mlp_bh]) > args.thresh / blk.mlp.k)
                    num_ffd_experts = int(choose.sum())

                    if num_ffd_experts < blk.mlp.k: # threshold too large
                        ind = np.argpartition(np.array(expert_usage[mlp_bh]), -blk.mlp.k)[-blk.mlp.k:]
                        choose[ind] = True
                        num_ffd_experts = blk.mlp.k
                    pruning_mlp[depth] = True
                else:
                    num_ffd_experts = blk.mlp.num_experts

            # print(args.the_task, depth, num_attn_experts, num_ffd_experts)
            # miss att_w_topk_loss
            the_blocks.append(
                MoEnhanceTaskBlock(
                        num_attn_experts=num_attn_experts, num_ffd_experts=num_ffd_experts,
                        drop_path=dpr[depth],
                        ffd_noise=self.R['ffd_noise'],
                        task_num=self.R['task_num'],
                        head_dim=self.R['head_dim'],
                        ffd_heads=self.R['ffd_heads'], 
                        noisy_gating=self.R['noisy_gating'],
                        dim=self.R['dim'], num_heads=self.R['num_heads'], mlp_ratio=self.R['mlp_ratio'], qkv_bias=self.R['qkv_bias'],
                        drop=dpr[depth], attn_drop=self.R['attn_drop'], norm_layer=self.R['norm_layer'],
                        moe_type=self.R['moe_type'],switchloss=self.R['switchloss'], zloss=self.R['zloss'], 
                        w_topk_loss=self.R['w_topk_loss'], limit_k=self.R['limit_k'],
                        post_layer_norm=self.R['post_layer_norm'],
                        use_moe_mlp=(self.R['twice_mlp']==False or (depth%2)==1),
                        use_moe_attn=(self.R['twice_attn']==False or (depth%2)==0),
                    )
                )

        del self.blocks
        self.blocks = nn.Sequential(*the_blocks)

        # Careful Here!!!
        # origin_task = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'segment_unsup2d', 'segment_unsup25d']
        # origin_task = args.ori_img_types
        origin_task = [type_ for type_ in args.ori_img_types if type_ != 'rgb']

        task_bh = -1
        for i, the_task in enumerate(origin_task):
            if the_task == args.the_task:
                task_bh = i 
                break
        assert task_bh >= 0

        checkpoint_all = torch.load(load_file, map_location='cpu')
        checkpoint = checkpoint_all['model']

        delete_key = []
        for c_key in checkpoint.keys():
            the_key = 'f_gate.' + str(task_bh) + '.'
            if ('f_gate.' in c_key) and (the_key not in c_key):
                # print('delete ', c_key)
                delete_key.append(c_key)
        for c_key in delete_key:
            del checkpoint[c_key]

        # print(checkpoint.keys())
        for depth, blk in enumerate(self.blocks):
            expert_usage = the_list[depth][args.the_task]

            prefix = 'blocks.' + str(depth) + '.attn.q_proj.'

            if task_bh != -1:
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.2.weight', prefix+'f_gate.'+'0'+'.2.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.2.bias', prefix+'f_gate.'+'0'+'.2.bias')

            # TaskMoe experts.w experts.b output_experts.w output_experts.b f_gate.task_bh.0

            # if hasattr(blk.attn, 'num_experts'):
            #     if blk.attn.num_experts > blk.attn.num_heads:
            # if pruning_attn:
            if pruning_attn[depth]:
                # select_id = (torch.from_numpy(np.array(expert_usage[0])) > args.thresh).nonzero().view(-1)

                select_id = (np.array(expert_usage[0]) > args.thresh / blk.attn.num_heads)
                num_attn_experts = int(select_id.sum())

                if num_attn_experts < blk.attn.num_heads: # threshold too large
                    ind = np.argpartition(np.array(expert_usage[0]), -blk.attn.num_heads)[-blk.attn.num_heads:]
                    select_id[ind] = True
                    num_attn_experts = blk.attn.num_heads
                select_id = torch.from_numpy(select_id).nonzero().view(-1)
                print('select_id: ', select_id)

                for words in ['experts.w', 'experts.b', 'output_experts.w', 'output_experts.b', 'f_gate.'+'0'+'.0.weight', 'f_gate.'+'0'+'.0.bias']:
                    the_key = prefix+words
                    if the_key in checkpoint:
                        # print(words, ' : ', checkpoint[the_key].shape)

                        if 'f_gate' not in the_key:
                            tgt_key = the_key
                        elif '.weight' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.weight'
                        elif '.bias' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.bias'

                        if blk.attn.q_proj.noisy_gating and 'f_gate' in words:
                            the_id = select_id + checkpoint[the_key].shape[0] // 2
                            the_id = torch.cat((select_id, the_id), 0)
                            # print('the_id: ', the_id, tgt_key)
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, the_id)
                        else:
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, select_id)
                            # print(tgt_key, select_id, checkpoint[tgt_key].shape)

            prefix = 'blocks.' + str(depth) + '.mlp.'
            if task_bh != -1:
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')
            

            # if pruning_mlp:
            # if hasattr(blk.mlp, 'num_experts'):
            #     if blk.mlp.num_experts > blk.mlp.k:
            if pruning_mlp[depth]:
                # select_id = (torch.from_numpy(np.array(expert_usage[mlp_bh])) > args.thresh).nonzero().view(-1)
                select_id = (np.array(expert_usage[mlp_bh]) > args.thresh / blk.mlp.k)
                num_ffd_experts = int(select_id.sum())

                if num_ffd_experts < blk.mlp.k: # threshold too large
                    ind = np.argpartition(np.array(expert_usage[mlp_bh]), -blk.mlp.k)[-blk.mlp.k:]
                    select_id[ind] = True
                    num_ffd_experts = blk.mlp.k
                select_id = torch.from_numpy(select_id).nonzero().view(-1)

                for words in ['experts.w', 'experts.b', 'output_experts.w', 'output_experts.b', 'f_gate.'+'0'+'.0.weight', 'f_gate.'+'0'+'.0.bias']:
                    the_key = prefix+words
                    if the_key in checkpoint:
                        # print(words, ' : ', checkpoint[the_key].shape)
                        # print('depth: ', depth, the_key, select_id)

                        if 'f_gate' not in the_key:
                            tgt_key = the_key
                        elif '.weight' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.weight'

                            # checkpoint[prefix+'f_gate.'+'0'+'.0.weight'] = checkpoint[prefix+'f_gate.'+str(task_bh)+'.0.weight']
                        elif '.bias' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.bias'

                            # checkpoint[prefix+'f_gate.'+'0'+'.0.bias'] = checkpoint[prefix+'f_gate.'+str(task_bh)+'.0.bias']

                        if blk.mlp.noisy_gating and 'f_gate' in words:
                            the_id = select_id + checkpoint[the_key].shape[0] // 2
                            the_id = torch.cat((select_id, the_id), 0)
                            # print('the_id: ', the_id, tgt_key)
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, the_id)
                            # print(checkpoint[tgt_key].shape)
                        else:
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, select_id)

        src_key = 'task_heads.' + str(task_bh) + '.'
        tgt_key = 'task_heads.0.'
        new_dict = {}
        delete_key = []
        for c_key in checkpoint.keys():
            if src_key in c_key:
                new_dict[tgt_key + c_key[len(src_key):]] = checkpoint[c_key]
                # print(c_key, tgt_key + c_key[len(src_key):])
            if 'task_heads' in c_key:
                delete_key.append(c_key)

        for c_key in delete_key:
            del checkpoint[c_key]
        checkpoint.update(new_dict)

        if task_bh != -1:
            checkpoint['task_embedding'] = checkpoint['task_embedding'][:,task_bh:task_bh+1]
        else:
            del checkpoint['task_embedding']

        return checkpoint

    def delete_ckpt(self, args): # the_task is not in origin task
        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            load_file = '/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'
        else:
            load_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'

        checkpoint_all = torch.load(load_file, map_location='cpu')
        checkpoint = checkpoint_all['model']

        delete_key = []
        for c_key in checkpoint.keys():
            if ('f_gate.' in c_key) or ('task_heads' in c_key):
                # print('delete ', c_key)
                delete_key.append(c_key)
        for c_key in delete_key:
            del checkpoint[c_key]

        del checkpoint['task_embedding']
        return checkpoint

    def frozen(self):
        self.patch_embed.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False
        for blk in self.blocks:
            blk.attn.kv_proj.requires_grad = False
            blk.attn.q_proj.experts.requires_grad = False
            blk.attn.q_proj.output_experts.requires_grad = False

            blk.mlp.experts.requires_grad = False
            blk.mlp.output_experts.requires_grad = False

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def get_zloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss

            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def get_topkloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss

            # break
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def all_clear(self):
        for blk in self.blocks:
            aux_loss = blk.attn.q_proj.init_aux_statistics()
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.init_aux_statistics()

    def visualize(self, vis_head=False, vis_mlp=False, model_name=''):
        all_list = []
        torch.set_printoptions(precision=2, sci_mode=False)

        for depth, blk in enumerate(self.blocks):
            layer_list = {}
            for i, the_type in enumerate(self.img_types):

                layer_list[the_type] = []
                if hasattr(blk.attn, 'num_experts'):
                    if blk.attn.num_experts > blk.attn.num_heads:
                        _sum = blk.attn.q_proj.task_gate_freq[i].sum()
                        layer_list[the_type].append((blk.attn.q_proj.task_gate_freq[i] / _sum * 100).tolist())
                        # print('L', depth, ' attn: ', blk.attn.q_proj.task_gate_freq[i] / _sum * 100)

                if hasattr(blk.mlp, 'num_experts'):
                    if blk.mlp.num_experts > blk.mlp.k:
                        _sum = blk.mlp.task_gate_freq[i].sum()
                        layer_list[the_type].append((blk.mlp.task_gate_freq[i] / _sum * 100).tolist())
                        # print('L', depth, ' mlp: ', blk.mlp.task_gate_freq[i] / _sum * 100)
            all_list.append(layer_list)
        print(all_list)

        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            torch.save(all_list, '/gpfs/u/home/AICD/AICDzich/scratch/' + str(model_name) + '_vis.t7')
        else:
            torch.save(all_list, '/gpfs/u/home/LMCG/LMCGzich/scratch/' + str(model_name) + '_vis.t7')

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # x = x + self.task_embedding[:,task_rank:task_rank+1, :]
        x_before = self.pos_drop(x)

        # apply Transformer blocks

        output = {}
        z_loss = 0
        for t, the_type in enumerate(self.img_types):
            x = x_before
            for blk in self.blocks:
                x = x + self.task_embedding[:, t:t+1, :]
                x, _ = blk(x, t)

            if 'class' in the_type:
                x = x[:, 1:, :].mean(dim=1)
                x = self.fc_norm(x)
            else:
                x = self.norm(x)
                x = x[:, 1:, :]
            output[the_type] = x
            if self.w_topk_loss > 0.0:
                z_loss = z_loss + self.get_topkloss()
        
        return output, z_loss

    def forward(self, x, task, get_flop=False, get_z_loss=False):

        output, z_loss = self.forward_features(x)
        for t, the_type in enumerate(self.img_types):
            output[the_type] = self.task_heads[t](output[the_type])

        if get_flop:
            return output['class_object']

        # self.all_clear()
        return output, z_loss + self.get_zloss()


# class MTVisionTransformerM3ViT(MTMobileViTMoETaskGating):
#     def __init__(self, img_types,
#                  **kwargs):
#         super(MTVisionTransformerM3ViT, self).__init__(img_types,**kwargs)

#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed

#         # x = x + self.task_embedding[:,task_rank:task_rank+1, :]
#         x_before = self.pos_drop(x)

#         # apply Transformer blocks

#         output = {}
#         z_loss = 0
#         for t, the_type in enumerate(self.img_types):
#             x = x_before
#             for blk in self.blocks:
#                 x = x + self.task_embedding[:, t:t+1, :]
#                 x, _ = blk(x, t)

#             if 'class' in the_type:
#                 x = x[:, 1:, :].mean(dim=1)
#                 x = self.fc_norm(x)
#             else:
#                 x = self.norm(x)
#                 x = x[:, 1:, :]
#             output[the_type] = x
#             z_loss = z_loss + self.get_topkloss()
#             z_loss = z_loss + self.get_zloss()
        
#         return output, z_loss

#     def forward(self, x, task, get_flop=False, get_z_loss=False):

#         output, z_loss = self.forward_features(x)
#         for t, the_type in enumerate(self.img_types):
#             output[the_type] = self.task_heads[t](output[the_type])

#         if get_flop:
#             return output['class_object']

#         return output, z_loss


def mt_m_vit_tiny(img_types, **kwargs): # 6.43M 
    model = MTMobileViT(img_types,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def mt_m_vit_small(img_types, **kwargs): # 23.48M 4.6G
    model = MTMobileViT(img_types,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_mlp16E4_small(img_types, **kwargs): # 67.37M 5.21G
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_768_12E3_8E2_small(img_types, **kwargs): # 5.17Gs bsz18
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=768, depth=12, num_heads=3, qkv_bias=True,
        num_attn_experts=12, head_dim=768//12 * 2,
        num_ffd_experts=8, ffd_heads=2, ffd_noise=True, mlp_ratio=1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# def m3vit_taskgate_mlp16E4_small(img_types, **kwargs): # 67.37M 5.21G
#     model = MTVisionTransformerM3ViT(img_types, 
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         num_attn_experts=6, head_dim=384//6 * 2,
#         num_ffd_experts=16, ffd_heads=4, ffd_noise=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

def mt_m_vit_topk_taskgate_mlp16E4_small(img_types, **kwargs): # 67.37M 5.21G
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2, w_topk_loss=0.1, limit_k=4, 
        num_ffd_experts=16, ffd_heads=4, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_topk_l6_taskgate_mlp16E4_small(img_types, **kwargs): # 67.37M 5.21G
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2, w_topk_loss=0.1, limit_k=6, 
        num_ffd_experts=16, ffd_heads=4, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_topk_taskgate_small_att(img_types, **kwargs): # 67.37M 5.21G
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3 * 2, head_dim=384//6 * 2, att_w_topk_loss=0.1, att_limit_k=6, 
        num_ffd_experts=2, ffd_heads=2, ffd_noise=True, w_topk_loss=0.0, limit_k=0, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_att(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 9, head_dim=384//6 * 2,
        num_ffd_experts=1, ffd_heads=1, ffd_noise=False, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_att_MI(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 9, head_dim=384//6 * 2,
        num_ffd_experts=1, ffd_heads=1, ffd_noise=False, mlp_ratio=4,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_att_MI_prob(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 9, head_dim=384//6 * 2,
        num_ffd_experts=1, ffd_heads=1, ffd_noise=False, mlp_ratio=4,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_att_MI_2(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 9, head_dim=384//6 * 2,
        num_ffd_experts=1, ffd_heads=1, ffd_noise=False, mlp_ratio=4,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_mlp16E4_small_MI(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.03, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_att_mlp_small_MI(img_types, **kwargs): # 33.25M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=12, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=1,
        w_MI=0.005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_att_mlp_small_MI_twice(img_types, **kwargs): # 34.17M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=15, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        twice_attn=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_mlp_small_MI(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=1,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_mlp_small_4_MI(img_types, **kwargs): # 32.64M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=8, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_att_mlp(img_types, **kwargs): # 
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 9, head_dim=384//6 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=True, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# (128 * 15 + 384 * 16) * 12 * 13 = 1.25M

def mt_m_vit_taskgate_small_att_mlp_MI(img_types, **kwargs): # 68.46M
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
        num_attn_experts=6 + 6, head_dim=384//6 * 2,
        num_ffd_experts=8, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.0005, switchloss=0.0, zloss=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_task0(img_types, **kwargs): # 60.17M 
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=4, ffd_heads=4, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_task0_MI(img_types, **kwargs): # 60.17M 
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2,
        num_ffd_experts=4, ffd_heads=4, ffd_noise=True,
        w_MI=0.00001, switchloss=0.0, zloss=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_taskgate_small_task2(img_types, **kwargs): # 60.17M 
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6 + 3 * 2, head_dim=384//6 * 2,
        num_ffd_experts=2 + 2 * 2, ffd_heads=2, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# def mt_m_vit_moa_small(img_types, **kwargs):
#     model = VisionTransformerMoA(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# def mt_m_vit_moe_small(img_types, **kwargs):
#     model = VisionTransformerMoE(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         num_attn_experts=6*8, head_dim=384//6 * 2,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


def mt_m_vit_taskgate_att_mlp_base_MI_twice(img_types, **kwargs): # number of params (M): 195.80
    model = MTMobileViTMoETaskGating(img_types, 
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
        num_attn_experts=24, head_dim=768//12 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        twice_attn=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mt_m_vit_base_patch16(img_types, **kwargs):
    model = MTMobileViT(img_types,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_base(img_types, **kwargs): # 89.42M 17.58G
    model = MTMobileViT(img_types,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mt_m_vit_large_patch16(img_types, **kwargs):
    model = MTMobileViT(img_types,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mt_m_vit_huge_patch14(img_types, **kwargs):
    model = MTMobileViT(img_types,
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model





if __name__ == '__main__':
    img_types = ['rgb', 'class_object', 'class_scene', 'depth_euclidean', 'normal', 'segment_semantic']
    norm_layer=torch.nn.Identity
    model = MTMobileViT(img_types=img_types, norm_layer=norm_layer)
    print(model)