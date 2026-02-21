import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jingjaijan/isaacgym/IsaacGymEnvs/assets/urdf/hanu_a3_description/install/hanu_a3_description'
