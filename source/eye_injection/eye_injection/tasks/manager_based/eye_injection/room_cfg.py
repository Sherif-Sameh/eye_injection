import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR


# Room parameters 
ROOM_SIZE = 5.0
ROOM_HEIGHT = 2.5
ROOM_THICKNESS = 0.1


# Room configuration
ROOM_CFG = {
    "Ground": RigidObjectCfg(
        prim_path="/World/Ground",
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

GROUND_TEXTURE_PATHS = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_grey.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_brick_tile.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_concrete_sidewalk_new.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_drywall_painted_aqua.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_plastic_blue.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_tile_square_green.jpg",
    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Patterns/nv_wood_boards_brown.jpg",
]