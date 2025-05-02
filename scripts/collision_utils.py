import numpy as np

# 精确的碰撞检测函数
def get_rotated_corners(position, length, width, heading):
    """
    获取车辆旋转后的四个顶点
    """
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])

    half_size = np.array([[ length/2,  width/2],
                          [ length/2, -width/2],
                          [-length/2, -width/2],
                          [-length/2,  width/2]])

    corners = (rot @ half_size.T).T + position
    return corners  # shape (4, 2)

def project_onto_axis(corners, axis):
    """
    投影点集到轴上，返回 min/max 投影值
    """
    projections = corners @ axis
    return projections.min(), projections.max()

def check_rotated_rectangle_collision(v1, v2):
    """
    判断两个带朝向的矩形是否碰撞（使用 SAT 分离轴定理）
    """
    corners1 = get_rotated_corners(v1.position, v1.LENGTH, v1.WIDTH, v1.heading)
    corners2 = get_rotated_corners(v2.position, v2.LENGTH, v2.WIDTH, v2.heading)

    # 获取 4 个轴（两个矩形的边法向）
    axes = []
    for corners in [corners1, corners2]:
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])  # 法向量
            axis = axis / np.linalg.norm(axis)
            axes.append(axis)

    # 在每个轴上投影两个矩形，如果有一个轴没有重叠，则说明不相交
    for axis in axes:
        min1, max1 = project_onto_axis(corners1, axis)
        min2, max2 = project_onto_axis(corners2, axis)
        if max1 < min2 or max2 < min1:
            return False  # 有分离轴 → 不相交

    return True  # 所有轴都有重叠 → 碰撞