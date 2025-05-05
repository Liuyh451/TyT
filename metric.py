import numpy as np


def compute_ape(pred_coords, true_coords, radius=6371.0):
    """
    计算 APE（Average Position Error）

    参数:
        pred_coords: ndarray, shape (n, 2), 预测的坐标 (lat, lon)
        true_coords: ndarray, shape (n, 2), 真实的坐标 (lat, lon)
        radius: float, 地球半径，单位：km（默认 6371 km）

    返回:
        ape: float, 平均位置误差 (km)
    """
    pred_lat = np.radians(pred_coords[:, 0])
    pred_lon = np.radians(pred_coords[:, 1])
    true_lat = np.radians(true_coords[:, 0])
    true_lon = np.radians(true_coords[:, 1])

    dlat = pred_lat - true_lat
    dlon = pred_lon - true_lon

    m1 = np.sin(dlat / 2) ** 2
    m2 = np.cos(true_lat) * np.cos(pred_lat) * (np.sin(dlon / 2) ** 2)
    a = m1 + m2
    c = 2 * np.arcsin(np.sqrt(a))

    distances = radius * c  # 每个点对之间的球面距离
    ape = np.mean(distances)

    return ape
