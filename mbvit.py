from timm.models import ByobNet
from timm.models.mobilevit import mobilevit_xxs, mobilevit_xs, mobilevit_s
from timm.models.mobilevit import mobilevitv2_050
class MobileViT(ByobNet):
    def __init__(self, **kwargs):
        pretrained = kwargs['pretrained']
        model_version = kwargs['mbvit_version']
        self.model_version = model_version
        base_model = None
        self.image_height = kwargs['image_height']
        self.image_width = kwargs['image_width']
        filttered_args = set([
            'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 'qkv_bias', 
            'drop_path_rate', 'num_classes', 'norm_layer', 'patch_size', 
            'attn_drop_rate', 'drop_rate', 'mbvit_version', 'pretrained',
            'encode_output_size', 'image_width', 'image_height'
        ])
        print(set(kwargs)-filttered_args)
        
        if model_version == 'mobilevit_xxs':
            base_model = mobilevit_xxs(pretrained)
        elif model_version == 'mobilevit_xs':
            base_model = mobilevit_xs(pretrained)
        elif model_version == 'mobilevit_s':
            base_model = mobilevit_s(pretrained)
        elif model_version == 'mobilevitv2_050':
            base_model = mobilevitv2_050(pretrained)
        else :
            assert False, f'unknown version of MobileViT: "{model_version}"'
        if base_model == None:
            assert False, f'model {model_version} initialization failed.'
        self.__dict__.update(base_model.__dict__)

    @classmethod
    def from_mobilevit_xxs(cls):
        return mobilevit_xxs()
    
    @classmethod
    def from_mobilevit_xs(cls):
        return mobilevit_xs()
    
    @classmethod
    def from_mobilevit_s(cls):
        return mobilevit_s()
    
    @classmethod
    def from_mobilevit_050(cls):
        return mobilevitv2_050()
    

if __name__ == '__main__':
    model = MobileViT(pretrained=False, mbvit_version='mobilevitv2_050')
    print(model)