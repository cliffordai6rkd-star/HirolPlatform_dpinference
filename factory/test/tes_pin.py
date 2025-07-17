import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(cur_path, "../../assets/franka_fr3/fr3_franka_hand.urdf")
    print(f'urdf path: {urdf_path}')

    mesh_dir = os.path.join(cur_path, "../../assets/franka_fr3")
    model, geo_model, colision_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir)
    data = model.createData()
    print("model name: " + model.name)
    print(f'nq: {model.nq}, nv: {model.nv}')
    total_Mass = pin.computeTotalMass(model, data)
    print('Total mass of the model: ', total_Mass)

    # Sample a random configuration
    q = pin.randomConfiguration(model)
    print(f"q: {q.T}")
    
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # Print out the placement of each joint of the kinematic tree
    # print(f'frame name: {model.frames[0]}')
    for name, oMi in zip(model.names, data.oMi):
        print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
    
    # for i in range(len(model.frames)):
    #     print(f'frame {i} name: {model.frames[i].name}, frame pose: {data.oMf[i]}')
    
    # Display the end-effector frame
    ee_frame_id = model.getFrameId("fr3_hand_tcp")
    print(f'ee frame id: {ee_frame_id}')
    print(f'ee frame name: {model.frames[ee_frame_id].name}')   
    
    
    M = pin.SE3(1)
    print(f'se3 example: {M}')
    
    rot = M.rotation
    quat = R.from_matrix(rot).as_quat()
    print(f'quat: {quat}')
    pin.computeJointJacobians(model, data, q)
        
    # dynamics param
    print(f'joint inertia: {data.M}')
    crba_M = pin.crba(model, data, q)
    print(f'crba joint inertia: {crba_M}')
    
    # Visualize the model using Meshcat
    viz = MeshcatVisualizer(model, colision_model, geo_model)
    viz.initViewer(open=False)
    viz.loadViewerModel()      
    viz.display(q)  
    
    import time
    time.sleep(10)