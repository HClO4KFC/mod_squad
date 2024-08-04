import os

root_dir = '/home/yhy/code/ckf/taskonomy'
# scenes_dict = {'train':[
#       'forkland', 'merom', 'klickitat', 'onaga', 'leonardo', 'marstons', 
#       'newfields', 'pinesdale', 'lakeville', 'cosmos', 'benevolence', 
#       'pomaria', 'tolstoy', 'shelbyville', 'allensville', 'wainscott', 
#       'beechwood', 'coffeen', 'stockman', 'hiteman', 'woodbine', 
#       'lindenwood', 'hanson', 'mifflinburg', 'ranchester'
#     ], 'val': ['wiconisco', 'corozal', 'collierville', 'markleeville', 'darden']
#     , 'test': ['ihlen', 'muleshoe', 'uvalda', 'noxapater', 'mcdade'] }
scenes_dict = {'train':['pomaria']}
tasks = ['rgb', 'class_object', 'class_scene', 'depth_euclidean', 'fragments', 'normal', 'segment_semantic']

if __name__ == '__main__':
    for part, scenes in scenes_dict.items():
        print(f'开始检查part {part},')
        for scene in scenes:
            # print(f'现在检查scene {scene}')
            prefixs = []
            ok = True
            for task in tasks:
                path = os.path.join(root_dir, part, task, 'taskonomy', scene)
                if not os.path.exists(path):
                    print(f'Warning: scene {scene} 缺少 {task} 分支')
                    ok=False
                    continue
                for file in os.listdir(path):
                    prefix = file.split('.')[0].split('domain')[0]
                    # print(f'{prefix}')
                    if task == 'rgb':
                        prefixs.append(prefix)
                    else:
                        if not prefix in prefixs:
                            print(f'Warning: scene {scene} path "{path}" file "{file}" 发现了独立的 {task} 分支')
                            ok=False
            if ok:
                print(f'scene {scene} 没有问题')
            else:
                print(f'scene {scene} 不完整')
                    
