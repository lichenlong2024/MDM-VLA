import numpy as np
import robosuite.utils.transform_utils as T


def get_scene_info(obs, env):
    """
    从环境对象中提取场景核心信息，返回机械臂、site、物体本体、物体关节的结构化字典。

    Args:
        env: LIBERO/Robosuite环境对象（需包含sim、obs、robots等核心属性）

    Returns:
        tuple: (robot_info, site_info, body_info, joint_info)
            - robot_info: 机械臂信息字典
            - site_info: site信息字典
            - body_info: 物体本体信息字典
            - joint_info: 环境中所有关节（含物体关节）的信息字典
    """
    # -------------------------- 1. 机械臂信息 (robot_info) --------------------------
    robot_info = {
        "joint_states": None,  # 关节位置
        "gripper_states": None,  # 夹爪位置
        "grasp_status": None,  # 抓取状态（注：需外部传入grasp，此处预留）
        "ee_pos": None,  # 末端位置（世界坐标系）
        "ee_quat": None,  # 末端姿态（四元数）
        "ee_states": None  # 末端状态（位置+轴角，与HDF5一致）
    }

    # 从obs提取机械臂核心数据（与HDF5格式对齐）
    # 关节位置
    if "robot0_joint_pos" in obs:
        robot_info["joint_states"] = obs["robot0_joint_pos"].round(4).tolist()
    # 夹爪位置
    if "robot0_gripper_qpos" in obs:
        robot_info["gripper_states"] = obs["robot0_gripper_qpos"].round(4).tolist()
    # 末端执行器位姿
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        ee_pos = obs["robot0_eef_pos"].round(4)
        ee_quat = obs["robot0_eef_quat"].round(4)
        ee_axisangle = T.quat2axisangle(ee_quat).round(4)
        ee_states = np.hstack((ee_pos, ee_axisangle)).tolist()
        robot_info["ee_pos"] = ee_pos.tolist()
        robot_info["ee_quat"] = ee_quat.tolist()
        robot_info["ee_states"] = ee_states

    # -------------------------- 2. Site信息 (site_info) --------------------------
    site_info = {}  # 键：site名称，值：{ "xpos": 世界坐标系位置 }
    sim = env.sim
    nsite = sim.model.nsite

    if nsite > 0:
        # 获取有效site名称（过滤空字符串）
        site_names = [sim.model.site_id2name(i) for i in range(nsite) if sim.model.site_id2name(i)]
        # 获取site实时绝对位置（世界坐标系）
        site_xpos = sim.data.site_xpos.round(4)  # shape: (nsite, 3)

        for name, xpos in zip(site_names, site_xpos):
            site_info[name] = {
                "xpos": xpos.tolist()  # [x, y, z]（单位：m）
            }

    # -------------------------- 3. 物体本体信息 (body_info) --------------------------
    body_info = {}  # 键：body名称，值：{ "xpos": 位置, "xquat": 姿态 }
    # 获取有效body名称（过滤空字符串）
    all_body_names = [name for name in sim.model.body_names if name]

    # 提取每个物体的实时位姿（世界坐标系）
    for name in all_body_names:
        try:
            body_id = sim.model.body_name2id(name)
            xpos = sim.data.body_xpos[body_id].round(4)  # 位置 [x,y,z]
            xquat = sim.data.body_xquat[body_id].round(4)  # 姿态 [qx,qy,qz,qw]
            body_info[name] = {
                "xpos": xpos.tolist(),
                "xquat": xquat.tolist()
            }
        except Exception as e:
            body_info[name] = {
                "xpos": None,
                "xquat": None,
                "error": f"获取失败: {str(e)[:20]}"
            }

    # -------------------------- 4. 物体关节信息 (joint_info) --------------------------
    joint_info = {}  # 键：关节名称，值：{ "angle": 关节角度（弧度） }
    njoint = sim.model.njnt  # 总关节数量（含机器人和环境物体）
    # 获取所有关节名称（过滤空字符串）
    joint_names = [sim.model.joint_id2name(i) for i in range(njoint) if sim.model.joint_id2name(i)]
    for name in joint_names:
        # 获取关节ID
        joint_id = sim.model.joint_name2id(name)
        # 获取关节在qpos中的索引范围（hinge关节通常只占1个索引）
        adr = sim.model.jnt_qposadr[joint_id]
        # 提取关节角度（qpos中存储的是关节位置，对hinge关节而言就是角度）
        angle = sim.data.qpos[adr].round(4).item()  # .item()转为标量
        joint_info[name] = {
            "angle": angle  # 单位：弧度
        }


    return robot_info, site_info, body_info, joint_info