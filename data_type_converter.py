import numpy as np
target_h=40
target_w=40

def center_crop(data, target_h, target_w):
    """
    中心裁剪或填充数据到指定尺寸（不足填充0，超过则裁剪）
    
    Args:
        data: numpy.ndarray - 输入数据数组，支持2D [h,w] 或3D [h,w,c] 格式
        target_h: int - 目标高度尺寸
        target_w: int - 目标宽度尺寸
    
    Returns:
        numpy.ndarray - 处理后的数组，尺寸为 [target_h, target_w] 或 [target_h, target_w, c]
    """
    # 获取原始尺寸并处理通道维度
    h, w = data.shape[0], data.shape[1]
    channels = data.shape[2] if len(data.shape) == 3 else None

    # ====================== 高度维度处理 ======================
    # 裁剪或填充高度方向
    if h > target_h:
        start_h = (h - target_h) // 2
        data = data[start_h:start_h+target_h, ...]
    elif h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = (target_h - h) - pad_top
        pad_width = ((pad_top, pad_bottom), (0, 0)) if channels is None else ((pad_top, pad_bottom), (0, 0), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)

    # ====================== 宽度维度处理 ======================
    # 裁剪或填充宽度方向
    if w > target_w:
        start_w = (w - target_w) // 2
        data = data[:, start_w:start_w+target_w, ...]
    elif w < target_w:
        pad_left = (target_w - w) // 2
        pad_right = (target_w - w) - pad_left
        pad_width = ((0, 0), (pad_left, pad_right)) if channels is None else ((0, 0), (pad_left, pad_right), (0, 0))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)

    return data

# ====================== 主数据处理流程 ======================
# 处理2001-2005年的数据文件
for i in range(2001, 2006):
    # 构建文件路径并加载数据
    path = r"E:\Dataset\ERA5\Extracted\500hPa\UWind" + str(i) + ".npy"
    old_data = np.load(path, allow_pickle=True)
    
    # 遍历数据并调整形状
    for j in range(len(old_data)):
        if old_data[j].shape != (40, 40):
            old_data[j] = center_crop(old_data[j], target_h, target_w)
    
    # 数据验证与类型转换
    for idx, d in enumerate(old_data):
        try:
            arr = np.array(d, dtype=np.float32)
            if arr.shape != (40, 40):
                print(f"[!] 第 {idx} 个数据形状异常: {arr.shape}")
        except Exception as e:
            print(f"[!] 第 {idx} 个数据转换失败: {type(d)}, 错误: {e}")
    
    # 最终数据转换与保存
    fixed_data = np.array([np.array(d, dtype=np.float32) for d in old_data])
    np.save(path, fixed_data)
    
    # 输出验证信息
    print(fixed_data.shape)     # 预期形状 (863, H, W)
    print(fixed_data.dtype)     # 预期数据类型 float32
