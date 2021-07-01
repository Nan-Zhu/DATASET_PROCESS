import numpy as np
import matplotlib.pyplot as plt

history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second
total_frames = history_frames + future_frames



def draw_trajectory(predicted_data, real_data):
    """
    draw predicted and real trajectory
    predicted_data: matrix (objects, total_frames, xy)
    real_data: matrix (objects, total_frames, xy)
    """


    mean_xy = (real_data[0][0])
    real_data = real_data - mean_xy
    predicted_data = predicted_data - mean_xy

    plt.figure()
    for i in predicted_data:
        x = i[:,0]
        y = i[:,1]
        plt.scatter(x, y,color='r',label='predicted')
    for i in real_data:
        x = i[:, 0]
        y = i[:, 1]
        plt.scatter(x, y, color='b', label='real')
    plt.show()

if __name__ == '__main__':
   xy_1 = np.array(
        [(1067.493, 959.049), (1066.889, 959.071), (1066.286, 959.094), (1065.682, 959.118), (1065.08, 959.142),
         (1064.478, 959.167),
         (1063.877, 959.192), (1063.279, 959.218), (1062.682, 959.245), (1062.087, 959.272), (1061.494, 959.301),
         (1060.903, 959.331)])

   xy_2 = np.array(
        [(1070.356, 945.323), (1072.136, 945.332), (1073.918, 945.348), (1075.702, 945.369), (1077.488, 945.398),
         (1079.275, 945.434),
         (1081.064, 945.478), (1082.854, 945.531), (1084.645, 945.593), (1086.437, 945.666), (1088.231, 945.749),
         (1090.027, 945.843)])

   real_data = np.array([xy_1, xy_2])
   xy_3 = np.array(
        [(1067.493, 959.049), (1066.889, 959.071), (1066.286, 959.094), (1065.682, 959.118), (1065.08, 959.142),
         (1064.478, 959.167),
         (1063.877, 960.192), (1063.279, 960.218), (1062.682, 960.245), (1062.087, 960.272), (1061.494, 960.301),
         (1060.903, 960.331)])

   xy_4 = np.array(
        [(1070.356, 945.323), (1072.136, 945.332), (1073.918, 945.348), (1075.702, 945.369), (1077.488, 945.398),
         (1079.275, 945.434),
         (1081.064, 946.478), (1082.854, 946.531), (1084.645, 946.593), (1086.437, 946.666), (1088.231, 946.749),
         (1090.027, 946.843)])
   predicted_data = np.array([xy_3, xy_4])
   draw_trajectory(predicted_data, real_data)
