from glob import glob
import random
import os



def get_data_list(
    data_dir: str, 
    seed: int, 
    support_set_number: int, 
    query_set_number: int, 
    development_set_number: int, 
    estimation_set_number: int,
    endwith: str = '*.h5',
):
    
    # check len of the data_dir
    assert len(data_dir) == \
        support_set_number+query_set_number+development_set_number+estimation_set_number, \
            "x should be an integer"

    random.seed(seed)
    data_dir_list = glob(os.path.join(data_dir, endwith))
    data_dir_list.sort()
    random.shuffle(data_dir_list)
    return (
        data_dir_list[:support_set_number],
        data_dir_list[support_set_number:support_set_number+query_set_number],
        data_dir_list[support_set_number+query_set_number:support_set_number+query_set_number+development_set_number],
        data_dir_list[-estimation_set_number:]
    )




if __name__ == "__main__":
    a,b, c, d = get_data_list('/home/xiangcen/SPRV_Brain/data/BraTs_H5', 25,
                              1000, 1000, 150, 101)



    print(len(os.listdir('/home/xiangcen/SPRV_Brain/data/BraTs_H5')))
    print(len(a), len(b), len(c), len(d))
    
    
    print(d[32])

