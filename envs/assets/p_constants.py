"""Pybullet constants.
"""

PS_LIST = None

def set_ps(ps_list):
    """Set global pybullet constants.
    """
    global PS_LIST  # pylint:disable=global-statement
    PS_LIST = ps_list


CUBE_URDF = """<robot name="cube">
<link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value="1.0"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="{0} {1} {2}"/>
      </geometry>
       <material name="white">
        <color rgba="{3} {4} {5} {6}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="{0} {1} {2}"/>
      </geometry>
    </collision>
  </link>
</robot>"""


CYLINDER_URDF = """<robot name="cylinder">
<link name="base_link">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder length="{1:.2f}" radius="{0:.2f}"/>
      </geometry>
      <material name="color">
        <color rgba="{2} {3} {4} {5}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <cylinder length="{1:.2f}" radius="{0:.2f}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>
  </robot>"""


TABLE_URDF = """<robot name="table">
<link name="world"/>
  <joint name="fixed" type="fixed">
    <parent link="world"/>
    <child link="baseLink"/>
    <origin xyz="0 0 0"/>
  </joint>
  <link name="baseLink">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0.6"/>
       <mass value="0"/>
       <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0.6"/>
      <geometry>
          <mesh filename="table.obj" scale="{0} {1} 0.08"/>
      </geometry>
   <material name="framemat0">
      <color
         rgba="{5} {5} {5} {2}" />
      </material>
    </visual>
    <visual>
      <origin rpy="0 0 0" xyz="-{3} -{4} 0.28"/>
      <geometry>
        <box size="0.05 0.05 0.56"/>
      </geometry>
  <material name="framemat0"/>
    </visual>
    <visual>
      <origin rpy="0 0 0" xyz="-{3} {4} 0.28"/>
      <geometry>
        <box size="0.05 0.05 0.56"/>
      </geometry>
  <material name="framemat0"/>
  </visual>
    <visual>
      <origin rpy="0 0 0" xyz="{3} -{4} 0.28"/>
      <geometry>
        <box size="0.05 0.05 0.56"/>
      </geometry>
  <material name="framemat0"/>
  </visual>
    <visual>
      <origin rpy="0 0 0" xyz="{3} {4} 0.28"/>
      <geometry>
        <box size="0.05 0.05 0.56"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0.6"/>
      <geometry>
      <box size="{0} {1} 0.08"/>
      </geometry>
    </collision>
  </link>
</robot>"""
