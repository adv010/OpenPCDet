import os
import pickle

# Directory containing the pickle files
pickle_dir = "/mnt/data/adat01/adv_OpenPCDet/output/kitti_models/pv_rcnn_ssl_60/tsne_3diou_secndstage_ep10_3256160/10ep_pkls"

# List all pickle files in the directory
pickle_files = [file for file in os.listdir(pickle_dir) if file.endswith(".pkl")]

# Initialize an empty dictionary to store aggregated data
combined_data = {}

# Iterate over each pickle file
for file in pickle_files:
    # Load the pickle file
    with open(os.path.join(pickle_dir, file), "rb") as f:
        data = pickle.load(f)
    # Combine the data from the current file with the aggregated data
    for key, value in data.items():
        if key not in combined_data:
            combined_data[key] = []
        combined_data[key].extend(value)

# Dump the combined data into a pickle file in the same directory
output_file = os.path.join(pickle_dir, "combined_ckpt10_22ep_tsnescores.pkl")
with open(output_file, "wb") as f:
    pickle.dump(combined_data, f)

print(f"Combined data has been dumped into '{output_file}'.")