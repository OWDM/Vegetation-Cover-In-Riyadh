import numpy as np
# Replace NaN values with a very small number
image_path = "C:\\Users\\Musae\\Documents\\GitHub-REPOs\\Senior-project-main\\Codes\\normal codes\\sub_images\\RUH_2018-12-15_0_0.npy"
ndvi_processed = np.nan_to_num(image_path, nan=0.01)