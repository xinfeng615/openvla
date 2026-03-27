import os
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ========== 核心修复：彻底禁用 Google Cloud 联网检查 ==========
tfds.core.utils.gcs_utils._is_gcs_disabled = True

# ========== 核心修复 2：类名显式加上下划线，强制 TFDS 保留 ==========
class MetaworldMl10_50e(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        # 严格按照 OpenVLA RLDS 标准定义数据结构
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_primary': tfds.features.Image(shape=(256, 256, 3), dtype=tf.uint8),
                        # 注意：这里已经被我们强制设定为 7 维
                        'state': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    }),
                    # 动作同样设定为 7 维
                    'action': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    'discount': tfds.features.Scalar(dtype=tf.float32),
                    'reward': tfds.features.Scalar(dtype=tf.float32),
                    'is_first': tfds.features.Scalar(dtype=tf.bool),
                    'is_last': tfds.features.Scalar(dtype=tf.bool),
                    'is_terminal': tfds.features.Scalar(dtype=tf.bool),
                    'language_instruction': tfds.features.Text(),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager):
        # 告诉构建器去哪里读取我们采集的 HDF5 文件
        return {'train': self._generate_examples(path='/root/autodl-tmp/metaworld_hdf5')}

    def _generate_examples(self, path):
        episode_id = 0
        for h5_file in os.listdir(path):
            if not h5_file.endswith('.hdf5'): continue
            file_path = os.path.join(path, h5_file)
            
            with h5py.File(file_path, 'r') as f:
                data_grp = f['data']
                for ep_key in data_grp.keys():
                    ep_grp = data_grp[ep_key]
                    images = ep_grp['image_primary'][:]
                    proprios = ep_grp['proprio'][:]
                    actions = ep_grp['action'][:]
                    instruction = ep_grp.attrs['language_instruction']
                    
                    episode_length = len(actions)
                    steps = []
                    
                    for i in range(episode_length):
                        # 【核心防爆逻辑】：4维补零扩充至7维
                        # 状态：[x, y, z] + [0, 0, 0] + [gripper]
                        padded_state = np.concatenate((proprios[i][:3], np.zeros(3), proprios[i][3:4]))
                        # 动作：[x, y, z] + [0, 0, 0] + [gripper]
                        padded_action = np.concatenate((actions[i][:3], np.zeros(3), actions[i][3:4]))
                        
                        steps.append({
                            'observation': {
                                'image_primary': images[i],
                                'state': padded_state.astype(np.float32),
                            },
                            'action': padded_action.astype(np.float32),
                            'discount': 1.0,
                            'reward': float(i == episode_length - 1),
                            'is_first': (i == 0),
                            'is_last': (i == episode_length - 1),
                            'is_terminal': (i == episode_length - 1),
                            'language_instruction': instruction,
                        })
                    
                    yield str(episode_id), {
                        'steps': steps,
                        'episode_metadata': {'file_path': file_path}
                    }
                    episode_id += 1

if __name__ == '__main__':
    # 手动实例化 Builder，注意这里调用的是新类名 MetaworldMl10_50e
    builder = MetaworldMl10_50e(data_dir='/root/autodl-tmp/tensorflow_datasets')
    
    print("🚀 开始构建 RLDS 数据集，请稍候...")
    # 启动转换与预处理过程
    builder.download_and_prepare()
    
    print("🎉 数据集构建完成！已保存至 /root/autodl-tmp/tensorflow_datasets/metaworld_ml10_50e")