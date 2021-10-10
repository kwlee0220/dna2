from dna.platform import CameraInfo, DNAPlatform
from dna.types import Size2i

platform = DNAPlatform()
platform.connect()

# camera_info_rset = platform.get_resource_set('camera_infos')
# print(camera_info_rset.get_all())

# info = CameraInfo("other:1", 1, Size2i([1024, 768]))
# camera_info_rset.insert(info)
# print(camera_info_rset.get_all())

trajectories = platform.get_resource_set('trajectories')
for traj in trajectories.get_where("path_count > 50"):
    print(traj)