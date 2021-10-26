import numpy as np


data_name_list = []
data_name_list.append("halfcheetah-random")
data_name_list.append("halfcheetah-medium")
data_name_list.append("halfcheetah-medium-replay")
data_name_list.append("halfcheetah-medium-expert")
data_name_list.append("hopper-random")
data_name_list.append("hopper-medium")
data_name_list.append("hopper-medium-replay")
data_name_list.append("hopper-medium-expert")
data_name_list.append("walker2d-random")
data_name_list.append("walker2d-medium")
data_name_list.append("walker2d-medium-replay")
data_name_list.append("walker2d-medium-expert")

for data_name in data_name_list:

    raw_score_mean = []
    raw_score_std  = []
    norm_score_mean = []
    norm_score_std  = []
    for alpha_str in ["0.0","0.2"]:
        temp_raw_score=[]
        for i in range(5):
            filename = "data/"+ data_name + "-v2/seed" + str(i+1) + "/progress_alpha" + alpha_str + "_iter1.csv"
            data = np.loadtxt(filename)
            temp_raw_score.append(data[-1,1])
        temp_raw_score = np.array(temp_raw_score)
        if "halfcheetah" in data_name:
            temp_norm_score = 100*(temp_raw_score + 280.178953) / (12135.0 + 280.178953)
        if "walker2d" in data_name:
            temp_norm_score = 100*(temp_raw_score - 1.629008) / (4592.3 - 1.629008)
        if "hopper" in data_name:
            temp_norm_score = 100*(temp_raw_score +20.272305) / (3234.3 +20.272305)

        raw_score_mean.append(temp_raw_score.mean())
        raw_score_std.append(temp_raw_score.std())
        norm_score_mean.append(temp_norm_score.mean())
        norm_score_std.append(temp_norm_score.std())

    #print(data_name,raw_score_mean)
    print(data_name,norm_score_mean,norm_score_std)

