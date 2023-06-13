import pickle
import random
random.seed(42)

data_file_path = "/mnt/data/adat01/adv_OpenPCDet/pcdet/utils/frame_splits/kitti_infos_train_37.pkl"
features_file_path = "/mnt/data/adat01/adv_OpenPCDet/pcdet/utils/frame_splits/point_features_37_unfiltered.pkl"
split_1_all_frames ={'006059', '000666', '003454', '005006', '005200', '005173', '004506', '001484', '001087', '005044', '006846', '001276', '004198', '006931', '001532', '000882', '000303', '002410', '005039', '001457', '006600', '004663', '004847', '000643', '006494', '002862', '007106', '006081', '002026', '000553', '000710', '005788', '000632', '006642', '005428', '001988','000585'}
print(len(split_1_all_frames))
num_car = 0
num_car_frames = 0
car_frame_ids = []
num_cyc = 0
num_cyc_frames = 0
cyc_frame_ids = []
num_ped = 0
num_ped_frames = 0
ped_frame_ids = []

with open(data_file_path, "rb") as file:
    data = pickle.load(file)
    car_dict = data['Car']
    for sample in car_dict:

        num_car += len([sample['name'] == 'Car'])
        car_frame_ids.append((sample['image_idx']))
        num_car_frames = len(set(car_frame_ids))

    cyc_dict = data['Cyclist']
    for sample in cyc_dict:
        num_cyc += len([sample['name'] == 'Cyclist'])
        cyc_frame_ids.append((sample['image_idx']))
        num_cyc_frames = len(set(cyc_frame_ids))

    ped_dict = data['Pedestrian']
    for sample in ped_dict:
        num_ped += len([sample['name'] == 'Pedestrian'])
        ped_frame_ids.append((sample['image_idx']))
        num_ped_frames = len(set(ped_frame_ids))


message =  f"For Split : {data_file_path}"
print(message)
message = f"There are {num_car} number of cars in {num_car_frames}/{37} frames"
message2 = f"{set(car_frame_ids)}"
print(message)
print(message2)
message = f"There are {num_cyc} number of cyclists in {num_cyc_frames}/{37} frames"
message2 = f"{set(cyc_frame_ids)}"
print(message)
print(message2)
message = f"There are {num_ped} number of peds in {num_ped_frames}/{37} frames"
message2 = f"{set(ped_frame_ids)}"
print(message)
print(message2)

car_cyc_ped_frames = [frame for frame in set(car_frame_ids) and  set(ped_frame_ids) and  set(cyc_frame_ids)]
print("Frames with all 3 classes :",len(car_cyc_ped_frames),car_cyc_ped_frames)

car_ped_frames =[frame for frame in set(car_frame_ids) and  set(ped_frame_ids) ]
print("Frames with car + ped :",len(car_ped_frames),car_ped_frames)

car_cyc_frames =[frame for frame in set(car_frame_ids) and  set(cyc_frame_ids) ]
print("Frames with car + cyc :",len(car_cyc_frames),car_cyc_frames)


cyc_ped_frames =[frame for frame in set(cyc_frame_ids) and  set(ped_frame_ids) ]
print("Frames with cyc +ped :",len(cyc_ped_frames),cyc_ped_frames)


#Generate 2 random splits
split1 = set(random.sample(split_1_all_frames,len(split_1_all_frames)//2))
split2 = split_1_all_frames - split1

print("TS:",split1)
print(len(split1))
print("TL :",split2)
print(len(split2))
