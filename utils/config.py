import yaml
import os



class MainConfig():


    def __init__(self, path: str, extra=None):
        assert os.path.exists(path), \
            'Config path ({}) does not exist'.format(path)
        assert path.endswith('yml') or path.endswith('yaml'), \
            'Config file ({}) should be yaml format'.format(path)
        args_dict = self.parse_from_yaml(path)
        

        self.add_nest_dict_item(args_dict)
        if extra:
            self.add_nest_dict_item(extra)
            

    def add_nest_dict_item(self, nest_dict: dict):
        for key, value in nest_dict.items():
            setattr(self, key, SubConfig(value))
            current_attr = getattr(self, key)
            for sub_key, sub_value in current_attr.sub_dict.items():
                setattr(current_attr, sub_key, sub_value)
                
                
    def add_dict_item(self, dict: dict):
        for key, value in dict.items():
            setattr(self, key, value)
            
            
    def save_config_file(self, file_dir: str):
        pass
        
            
        



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
    cfg = MainConfig('/home/xiangcen/SPRV_Brain/configs/main.yml', {'others': {'device': 'cuda:0'}})
    for i in cfg.workflow_evaluation.t_values:
        print(i)
    print(type(cfg.workflow_evaluation.pre_train))
    print(cfg.others.device)
    
    cfg.add_dict_item({'random': [12, 32, 43]})
    print(cfg.random)