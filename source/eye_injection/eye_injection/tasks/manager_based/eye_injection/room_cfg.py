import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


# Room parameters 
ROOM_SIZE = 5.0
ROOM_HEIGHT = 2.5
ROOM_THICKNESS = 0.1


# Room configuration
ROOM_CFG = {
    "Floor": RigidObjectCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_SIZE, ROOM_SIZE, ROOM_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.7),
                roughness=0.2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, -ROOM_THICKNESS / 2)
        ),
    ),
    "Wall_PosX": RigidObjectCfg(
        prim_path="/World/Wall_PosX",
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_THICKNESS, ROOM_SIZE, ROOM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(ROOM_SIZE / 2, 0.0, ROOM_HEIGHT / 2)
        ),
    ),
    "Wall_NegX": RigidObjectCfg(
        prim_path="/World/Wall_NegX",
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_THICKNESS, ROOM_SIZE, ROOM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-ROOM_SIZE / 2, 0.0, ROOM_HEIGHT / 2)
        ),
    ),
    "Wall_PosY": RigidObjectCfg(
        prim_path="/World/Wall_PosY",
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_SIZE, ROOM_THICKNESS, ROOM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, ROOM_SIZE / 2, ROOM_HEIGHT / 2)
        ),
    ),
    "Wall_NegY": RigidObjectCfg(
        prim_path="/World/Wall_NegY",
        spawn=sim_utils.CuboidCfg(
            size=(ROOM_SIZE, ROOM_THICKNESS, ROOM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -ROOM_SIZE / 2, ROOM_HEIGHT / 2)
        ),
    ), 
}
