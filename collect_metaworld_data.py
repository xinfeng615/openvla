import os
import numpy as np
import metaworld
import h5py
import metaworld.policies as policies
from experiments.robot.metaworld.metaworld_env import MetaworldEnv

# 映射 ML10 任务名称到新版 V3 专家策略类
POLICY_MAP = {
    'reach-v3': 'SawyerReachV3Policy',
    'push-v3': 'SawyerPushV3Policy',
    'pick-place-v3': 'SawyerPickPlaceV3Policy',
    'door-open-v3': 'SawyerDoorOpenV3Policy',
    'drawer-close-v3': 'SawyerDrawerCloseV3Policy',
    'button-press-topdown-v3': 'SawyerButtonPressTopdownV3Policy',
    'peg-insert-side-v3': 'SawyerPegInsertionSideV3Policy',
    'window-open-v3': 'SawyerWindowOpenV3Policy',
    'sweep-v3': 'SawyerSweepV3Policy',
    'basketball-v3': 'SawyerBasketballV3Policy',
}

def collect_ml10_data(episodes_per_task=50, max_steps=500):
    # 将生成的 HDF5 文件保存在数据盘
    save_dir = "/root/autodl-tmp/metaworld_hdf5"
    os.makedirs(save_dir, exist_ok=True)
    
    ml10 = metaworld.ML10()
    
    for task_name in ml10.train_classes.keys():
        print(f"\n🚀 开始采集任务: {task_name}")
        env = MetaworldEnv(task_name)
        
        base_name = task_name.replace('-goal-observable', '')
        
        # 获取对应的专家策略
        policy_class_name = POLICY_MAP.get(base_name)
        if not policy_class_name:
            print(f"⚠️ 字典中未定义 {base_name} 的策略，跳过...")
            continue
            
        try:
            policy_cls = getattr(policies, policy_class_name)
            policy = policy_cls()
        except AttributeError:
            print(f"❌ 找不到策略 {policy_class_name}，跳过 {task_name}")
            continue
            
        task_instruction = env.get_task()["language_instruction"][0]
        
        h5_path = os.path.join(save_dir, f"{base_name}.hdf5")
        with h5py.File(h5_path, "w") as f:
            data_grp = f.create_group("data")
            
            success_count = 0
            while success_count < episodes_per_task:
                obs_dict, info = env.reset()
                images, proprios, actions = [], [], []
                episode_success = False
                
                for step in range(max_steps):
                    # 1. 记录相机图像和本体状态
                    images.append(obs_dict["image_primary"])
                    proprios.append(obs_dict["proprio"])
                    
                    # 2. 从环境的 info 字典中提取专家需要的原始 39 维物理状态，并获取完美动作
                    raw_state = info["state"]
                    action = policy.get_action(raw_state)
                    actions.append(action)
                    
                    # 3. 步进环境
                    obs_dict, reward, done, trunc, info = env.step(action)
                    
                    if info.get('success', False):
                        episode_success = True
                        
                    if done or trunc:
                        break
                        
                # 只有当任务真正成功时，才将这段轨迹写入文件
                if episode_success:
                    ep_grp = data_grp.create_group(f"demo_{success_count}")
                    ep_grp.create_dataset("image_primary", data=np.array(images, dtype=np.uint8))
                    ep_grp.create_dataset("proprio", data=np.array(proprios, dtype=np.float32))
                    ep_grp.create_dataset("action", data=np.array(actions, dtype=np.float32))
                    ep_grp.attrs["language_instruction"] = task_instruction
                    
                    success_count += 1
                    print(f"  ✅ 成功收集轨迹: {success_count}/{episodes_per_task} (耗时步数: {len(actions)})")
                else:
                    # 专家策略在某些随机初始化下也会失误，失败的直接丢弃
                    print(f"  🔄 专家策略失误，重新尝试本回合...")
                    
        print(f"🎉 任务 {task_name} 数据已保存至 {h5_path}")

if __name__ == "__main__":
    collect_ml10_data()
