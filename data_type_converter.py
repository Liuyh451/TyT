# 假设你能加载旧数据为 old_data
import numpy as np
target_h=40
target_w=40
def center_crop(data, target_h, target_w):
    """
    中心裁剪或填充到指定尺寸（不足填充0，超过裁剪）
    :param data: 输入数据（支持2D数组 [h,w] 或3D数组 [h,w,c]）
    :param target_h: 目标高度
    :param target_w: 目标宽度
    :return: 处理后的数据（尺寸为 [target_h, target_w] 或 [target_h, target_w, c]）
    """
    # 获取当前数据的尺寸（支持2D和3D输入）
    h, w = data.shape[0], data.shape[1]
    channels = data.shape[2] if len(data.shape) == 3 else None  # 处理通道维度

    # ------------------------------ 高度处理 ------------------------------
    if h > target_h:  # 高度超过目标：中心裁剪
        start_h = (h - target_h) // 2
        end_h = start_h + target_h
        data = data[start_h:end_h, ...]
    elif h < target_h:  # 高度不足：上下填充0（中心填充）
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        if channels is None:  # 2D数据
            data = np.pad(data, pad_width=((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
        else:  # 3D数据（保留通道维度）
            data = np.pad(data, pad_width=((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # ------------------------------ 宽度处理 ------------------------------
    if w > target_w:  # 宽度超过目标：中心裁剪
        start_w = (w - target_w) // 2
        end_w = start_w + target_w
        data = data[:, start_w:end_w, ...]
    elif w < target_w:  # 宽度不足：左右填充0（中心填充）
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if channels is None:  # 2D数据
            data = np.pad(data, pad_width=((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
        else:  # 3D数据（保留通道维度）
            data = np.pad(data, pad_width=((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return data
for i in range(2001, 2006):
    path = r"E:\Dataset\ERA5\Extracted\500hPa\UWind" + str(i) + ".npy"
    old_data=np.load(path, allow_pickle=True)
    lenth = len(old_data)
    for j in range(lenth):
        if  old_data[j].shape != (40, 40):
            old_data[j]=center_crop(old_data[j], target_h, target_w)
    for i, d in enumerate(old_data):
        try:
            arr = np.array(d, dtype=np.float32)
            if arr.shape != (40, 40):
                print(f"[!] 第 {i} 个数据形状异常: {arr.shape}")
        except Exception as e:
            print(f"[!] 第 {i} 个数据转换失败: {type(d)}, 错误: {e}")

    # 转换成 float32 并堆叠
    fixed_data = np.array([np.array(d, dtype=np.float32) for d in old_data])

    print(fixed_data.shape)     # 应该是 (863, H, W)
    print(fixed_data.dtype)     # 应该是 float32
    np.save(path, fixed_data)






