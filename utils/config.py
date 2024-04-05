import yaml
import os



class MainConfig():


    def __init__(self, path: str, extra=None):
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)
        self.args_dict = self.parse_from_yaml(path)
        
        
        self.add_nest_dict_item(self.args_dict)
        if extra:
            self.add_nest_dict_item(extra)
            

    def add_nest_dict_item(self, nest_dict: dict):
        for key, value in nest_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SubConfig(value))
                current_attr = getattr(self, key)
                for sub_key, sub_value in current_attr.sub_dict.items():
                    setattr(current_attr, sub_key, sub_value)
            else:
                setattr(self, key, value)
                    
                
    def add_dict_item(self, dict: dict):
        for key, value in dict.items():
            setattr(self, key, value)
            

    @classmethod
    def parse_from_yaml(cls, path: str):
        """Parse a yaml file and build config"""
        with open(path, 'r') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)
        return dic
    


class SubConfig():
    
    def __init__(self, sub_dict: dict):
        self.sub_dict = sub_dict




if __name__ =="__main__":
    cfg = MainConfig('/home/xiangcen/SPRV_Brain/configs/main.yml')
    for i in cfg.workflow_evaluation.t_values:
        print(i)
    print(type(cfg.workflow_evaluation.pre_train))
    print(cfg.others.device)


    print(type(cfg.segmentation.learning_rate))
    
