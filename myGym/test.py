import gym
from myGym import envs
import cv2
import os, imageio
import numpy as np
from numpy import matrix
import pybullet as p
import pybullet_data
import pkg_resources
import commentjson
import argparse
import importlib
import myGym


clear = lambda: os.system('clear')
MODULE_PATH = importlib.util.find_spec(myGym.__name__).submodule_search_locations[0]


def task_objects_replacement(task_objects_new, task_objects_old, task_type):
    """
    If task_objects is given as a parameter, this method converts string into a proper format depending on task_type (null init for task_type reach)

    [{"init":{"obj_name":"null"}, "goal":{"obj_name":"cube_holes","fixed":1,"rand_rot":0, "sampling_area":[-0.5, 0.2, 0.3, 0.6, 0.1, 0.4]}}]
    """
    ret = copy.deepcopy(task_objects_old)
    if len(task_objects_new) > len(task_objects_old):
        msg = "More objects given than there are subtasks."
        raise Exception(msg)
    dest = "" #init or goal
    if task_type == "reach":
        dest = "goal"
    else:
        dest = "init"
    for i in range(len(task_objects_new)):
        ret[i][dest]["obj_name"] = task_objects_new[i]
    return ret


def get_arguments(parser):
    args = parser.parse_args()
    with open(args.config, "r") as f:
        arg_dict = commentjson.load(f)
    for key, value in vars(args).items():
        if value is not None and key != "config":
            if key in ["robot_init"]:
                arg_dict[key] = [float(arg_dict[key][i]) for i in range(len(arg_dict[key]))]
            elif key in ["task_objects"]:
                arg_dict[key] = task_objects_replacement(value, arg_dict[key], arg_dict["task_type"])
            else:
                arg_dict[key] = value
    return arg_dict


def get_parser():
    parser = argparse.ArgumentParser()
    # Envinronment
    parser.add_argument("-cfg", "--config", default=os.path.join(MODULE_PATH, "./configs/train_reach.json"), help="Can be passed instead of all arguments")
    parser.add_argument("-n", "--env_name", type=str, help="The name of environment")
    parser.add_argument("-ws", "--workspace", type=str, help="The name of workspace")
    parser.add_argument("-p", "--engine", type=str,  help="Name of the simulation engine you want to use")
    parser.add_argument("-sd", "--seed", type=int, help="Seed number")
    parser.add_argument("-d", "--render", type=str,  help="Type of rendering: opengl, opencv")
    parser.add_argument("-c", "--camera", type=int, help="The number of camera used to render and record")
    parser.add_argument("-vi", "--visualize", type=int,  help="Whether visualize camera render and vision in/out or not: 1 or 0")
    parser.add_argument("-vg", "--visgym", type=int,  help="Whether visualize gym background: 1 or 0")
    parser.add_argument("-g", "--gui", type=int, help="Wether the GUI of the simulation should be used or not: 1 or 0")
    # Robot
    parser.add_argument("-b", "--robot", type=str, default="kuka", help="Robot to train: kuka, panda, jaco ...")
    parser.add_argument("-bi", "--robot_init", nargs="*", type=float, help="Initial robot's end-effector position")
    parser.add_argument("-ba", "--robot_action", type=str, help="Robot's action control: step - end-effector relative position, absolute - end-effector absolute position, joints - joints' coordinates")
    parser.add_argument("-mv", "--max_velocity", type=float, help="Maximum velocity of robotic arm")
    parser.add_argument("-mf", "--max_force", type=float, help="Maximum force of robotic arm")
    parser.add_argument("-ar", "--action_repeat", type=int, help="Substeps of simulation without action from env")
    # Task
    parser.add_argument("-tt", "--task_type", type=str,  help="Type of task to learn: reach, push, throw, pick_and_place")
    parser.add_argument("-to", "--task_objects", nargs="*", type=str, help="Object (for reach) or a pair of objects (for other tasks) to manipulate with")
    parser.add_argument("-u", "--used_objects", nargs="*", type=str, help="List of extra objects to randomly appear in the scene")
    # Distractors
    parser.add_argument("-di", "--distractors", type=str, help="Object (for reach) to evade")
    parser.add_argument("-dm", "--distractor_moveable", type=int, help="can distractor move (0/1)")
    parser.add_argument("-ds", "--distractor_constant_speed", type=int, help="is speed of distractor constant (0/1)")
    parser.add_argument("-dd", "--distractor_movement_dimensions", type=int, help="in how many directions can the distractor move (1/2/3)")
    parser.add_argument("-de", "--distractor_movement_endpoints", nargs="*", type=float, help="2 coordinates (starting point and ending point)")
    parser.add_argument("-no", "--observed_links_num", type=int, help="number of robot links in observation space")
    #Reward
    parser.add_argument("-re", "--reward", type=str,  help="Defines how to compute the reward")
    parser.add_argument("-dt", "--distance_type", type=str, help="Type of distance metrics: euclidean, manhattan")
    #Train
    parser.add_argument("-w", "--train_framework", type=str,  help="Name of the training framework you want to use: {tensorflow, pytorch}")
    parser.add_argument("-a", "--algo", type=str,  help="The learning algorithm to be used (ppo2 or her)")
    parser.add_argument("-s", "--steps", type=int, help="The number of steps to train")
    parser.add_argument("-ms", "--max_episode_steps", type=int,  help="The maximum number of steps per episode")
    parser.add_argument("-ma", "--algo_steps", type=int,  help="The number of steps per for algo training (PPO2,A2C)")
    #Evaluation
    parser.add_argument("-ef", "--eval_freq", type=int,  help="Evaluate the agent every eval_freq steps")
    parser.add_argument("-e", "--eval_episodes", type=int,  help="Number of episodes to evaluate performance of the robot")
    #Saving and Logging
    parser.add_argument("-l", "--logdir", type=str,  help="Where to save results of training and trained models")
    parser.add_argument("-r", "--record", type=int, help="1: make a gif of model perfomance, 2: make a video of model performance, 0: don't record")
    #Mujoco
    parser.add_argument("-i", "--multiprocessing", type=int,  help="True: multiprocessing on (specify also the number of vectorized environemnts), False: multiprocessing off")
    parser.add_argument("-v", "--vectorized_envs", type=int,  help="The number of vectorized environments to run at once (mujoco multiprocessing only)")
    #Paths
    parser.add_argument("-m", "--model_path", type=str, help="Path to the the trained model to test")
    parser.add_argument("-vp", "--vae_path", type=str, help="Path to a trained VAE in 2dvu reward type")
    parser.add_argument("-yp", "--yolact_path", type=str, help="Path to a trained Yolact in 3dvu reward type")
    parser.add_argument("-yc", "--yolact_config", type=str, help="Path to saved config obj or name of an existing one in the data/Config script (e.g. 'yolact_base_config') or None for autodetection")
    parser.add_argument('-ptm', "--pretrained_model", type=str, help="Path to a model that you want to continue training")
    #Language
    # parser.add_argument("-nl", "--natural_language", type=str, default="",
    #                     help="If passed, instead of training the script will produce a natural language output "
    #                          "of the given type, save it to the predefined file (for communication with other scripts) "
    #                          "and exit the program (without the actual training taking place). Expected values are \"description\" "
    #                          "(generate a task description) or \"new_tasks\" (generate new tasks)")
    return parser


def visualize_sampling_area(arg_dict):
    rx = (arg_dict["task_objects"][0]["goal"]["sampling_area"][0] - arg_dict["task_objects"][0]["goal"]["sampling_area"][1])/2
    ry = (arg_dict["task_objects"][0]["goal"]["sampling_area"][2] - arg_dict["task_objects"][0]["goal"]["sampling_area"][3])/2
    rz = (arg_dict["task_objects"][0]["goal"]["sampling_area"][4] - arg_dict["task_objects"][0]["goal"]["sampling_area"][5])/2

    visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[rx,ry,rz], rgbaColor=[1,0,0,.2])
    collision = -1

    sampling = p.createMultiBody(
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=[arg_dict["task_objects"][0]["goal"]["sampling_area"][0]-rx, arg_dict["task_objects"][0]["goal"]["sampling_area"][2]-ry,arg_dict["task_objects"][0]["goal"]["sampling_area"][4]-rz],
    )


def visualize_trajectories(info, action):
    visualo = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0,0,1,.3])
    collision = -1
    p.createMultiBody(
            baseVisualShapeIndex=visualo,
            baseCollisionShapeIndex=collision,
            baseMass=0,
            basePosition=info['o']['actual_state'],
    )

    #visualr = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0,1,0,.5])
    #p.createMultiBody(
    #        baseVisualShapeIndex=visualr,
    #        baseCollisionShapeIndex=collision,
    #        baseMass=0,
    #        basePosition=info['o']['additional_obs']['endeff_xyz'],
    #)

    visuala = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1,0,0,.3])
    p.createMultiBody(
            baseVisualShapeIndex=visuala,
            baseCollisionShapeIndex=collision,
            baseMass=0,
            basePosition=action[:3],
    )


def visualize_goal(info):
    visualg = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[1,0,0,.5])
    collision = -1
    p.createMultiBody(
        baseVisualShapeIndex=visualg,
        baseCollisionShapeIndex=collision,
        baseMass=0,
        basePosition=info['o']['goal_state'],
    )


def change_dynamics(cubex,lfriction,rfriction,ldamping,adamping):
    p.changeDynamics(cubex, -1, lateralFriction=p.readUserDebugParameter(lfriction))
    p.changeDynamics(cubex,-1,rollingFriction=p.readUserDebugParameter(rfriction))
    p.changeDynamics(cubex, -1, linearDamping=p.readUserDebugParameter(ldamping))
    p.changeDynamics(cubex, -1, angularDamping=p.readUserDebugParameter(adamping))

    #visualrobot = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=1, rgbaColor=[0,1,0,.2])
    #collisionrobot = -1
    #sampling = p.createMultiBody(
    #    baseVisualShapeIndex=visualrobot,
    #    baseCollisionShapeIndex=collisionrobot,
    #    baseMass=0,
    #    basePosition=[0,0,0.3],
    #)


def visualize_infotext(action, env, info):
    p.addUserDebugText(f"Episode:{env.env.episode_number}",
        [.65, 1., 0.45], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, .3])
    p.addUserDebugText(f"Step:{env.env.episode_steps}",
        [.67, 1, .40], textSize=1.0, lifeTime=0.5, textColorRGB=[0.2, 0.8, 1])
    p.addUserDebugText(f"Subtask:{env.env.task.current_task}",
        [.69, 1, 0.35], textSize=1.0, lifeTime=0.5, textColorRGB=[0.4, 0.2, 1])
    p.addUserDebugText(f"Network:{env.env.reward.current_network}",
        [.71, 1, 0.3], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Action (Gripper):{matrix(np.around(np.array(action),2))}",
        [.73, 1, 0.25], textSize=1.0, lifeTime=0.5, textColorRGB=[1, 0, 0])
    p.addUserDebugText(f"Actual_state:{matrix(np.around(np.array(env.env.observation['task_objects']['actual_state'][:3]),2))}",
        [.75, 1, 0.2], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"End_effector:{matrix(np.around(np.array(env.env.robot.end_effector_pos),2))}",
        [.77, 1, 0.15], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 1, 0.0])
    p.addUserDebugText(f"        Object:{matrix(np.around(np.array(info['o']['actual_state']),2))}",
        [.8, 1, 0.10], textSize=1.0, lifeTime=0.5, textColorRGB=[0.0, 0.0, 1])
    p.addUserDebugText(f"Velocity:{env.env.max_velocity}",
        [.79, 1, 0.05], textSize=1.0, lifeTime=0.5, textColorRGB=[0.6, 0.8, .3])
    p.addUserDebugText(f"Force:{env.env.max_force}",
        [.81, 1, 0.00], textSize=1.0, lifeTime=0.5, textColorRGB=[0.3, 0.2, .4])


def detect_key(keypress, arg_dict, action):
    if 97 in keypress.keys() and keypress[97] == 1:
        action[2] += .03
        print(action)
    if 122 in keypress.keys() and keypress[122] == 1:
        action[2] -= .03
        print(action)
    if 65297 in keypress.keys() and keypress[65297] == 1:
        action[1] -= .03
        print(action)
    if 65298 in keypress.keys() and keypress[65298] == 1:
        action[1] += .03
        print(action)
    if 65295 in keypress.keys() and keypress[65295] == 1:
        action[0] += .03
        print(action)
    if 65296 in keypress.keys() and keypress[65296] == 1:
        action[0] -= .03
        print(action)
    if 120 in keypress.keys() and keypress[120] == 1:
        action[3] -= .03
        action[4] -= .03
        print(action)
    if 99 in keypress.keys() and keypress[99] == 1:
        action[3] += .03
        action[4] += .03
        print(action)
    if "step" in arg_dict["robot_action"]:
        action[:3] = np.multiply(action[:3], 10)
    elif "joints" in arg_dict["robot_action"]:
        print("Robot action: Joints - KEYBOARD CONTROL UNDER DEVELOPMENT")
        quit()
    #for i in range (env.action_space.shape[0]):
    #    env.env.robot.joints_max_velo[i] = p.readUserDebugParameter(maxvelo)
    #    env.env.robot.joints_max_force[i] = p.readUserDebugParameter(maxforce)
    return action


def test_env(env, arg_dict):
    env.reset()
    #arg_dict["gui"] = 1
    spawn_objects = False
    env.render("human")
    #env.reset()
    #Prepare names for sliders
    joints = ['Joint1','Joint2','Joint3','Joint4','Joint5','Joint6','Joint7','Joint 8','Joint 9', 'Joint10', 'Joint11','Joint12','Joint13','Joint14','Joint15','Joint16','Joint17','Joint 18','Joint 19']
    jointparams = ['Jnt1','Jnt2','Jnt3','Jnt4','Jnt5','Jnt6','Jnt7','Jnt 8','Jnt 9', 'Jnt10', 'Jnt11','Jnt12','Jnt13','Jnt14','Jnt15','Jnt16','Jnt17','Jnt 18','Jnt 19']
    cube = ['Cube1','Cube2','Cube3','Cube4','Cube5','Cube6','Cube7','Cube8','Cube9','Cube10','Cube11','Cube12','Cube13','Cube14','Cube15','Cube16','Cube17','Cube18','Cube19']
    cubecount = 0

    if arg_dict["gui"] == 0:
        print ("Add --gui 1 parameter to visualize environment")
        quit()

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    #newobject = p.loadURDF("cube.urdf", [3.1,3.7,0.1])
    #p.changeDynamics(newobject, -1, lateralFriction=1.00)
    #p.setRealTimeSimulation(1)
    if arg_dict["control"] == "slider":
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        if "joints" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i], env.env.robot.init_joint_poses[i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], env.action_space.low[i], env.action_space.high[i], .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i],env.action_space.low[i], env.action_space.high[i], env.env.robot.init_joint_poses[i])
        elif "absolute" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, arg_dict["robot_init"][i])
        elif "step" in arg_dict["robot_action"]:
            if 'gripper' in arg_dict["robot_action"]:
                print ("gripper is present")
                for i in range (env.action_space.shape[0]):
                    if i < (env.action_space.shape[0] - len(env.env.robot.gjoints_rest_poses)):
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)
                    else:
                        joints[i] = p.addUserDebugParameter(joints[i], -1, 1, .02)
            else:
                for i in range (env.action_space.shape[0]):
                    joints[i] = p.addUserDebugParameter(joints[i], -1, 1, 0)


    #maxvelo = p.addUserDebugParameter("Max Velocity", 0.1, 50, env.env.robot.joints_max_velo[0])
    #maxforce = p.addUserDebugParameter("Max Force", 0.1, 300, env.env.robot.joints_max_force[0])
    lfriction = p.addUserDebugParameter("Lateral Friction", 0, 100, 0)
    rfriction = p.addUserDebugParameter("Spinning Friction", 0, 100, 0)
    ldamping = p.addUserDebugParameter("Linear Damping", 0, 100, 0)
    adamping = p.addUserDebugParameter("Angular Damping", 0, 100, 0)
            #action.append(jointparams[i])
    if arg_dict["vsampling"] == True:
        visualize_sampling_area(arg_dict)

    #visualgr = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=.005, rgbaColor=[0,0,1,.1])

    if arg_dict["control"] == "random":
            action = env.action_space.sample()
    if arg_dict["control"] == "keyboard":
        action = arg_dict["robot_init"]
        if "gripper" in arg_dict["robot_action"]:
            action.append(.1)
            action.append(.1)
    if arg_dict["control"] == "slider":
        action = []
        for i in range (env.action_space.shape[0]):
            jointparams[i] = p.readUserDebugParameter(joints[i])
            action.append(jointparams[i])

    for e in range(50):
        env.reset()


        for t in range(arg_dict["max_episode_steps"]):


            if arg_dict["control"] == "slider":
                action = []
                for i in range (env.action_space.shape[0]):
                    jointparams[i] = p.readUserDebugParameter(joints[i])
                    action.append(jointparams[i])
                    #env.env.robot.joints_max_velo[i] = p.readUserDebugParameter(maxvelo)
                    #env.env.robot.joints_max_force[i] = p.readUserDebugParameter(maxforce)


            if arg_dict["control"] == "observation":
                if t == 0:
                    action = env.action_space.sample()
                else:

                    if "joints" in arg_dict["robot_action"]:
                        action = info['o']["additional_obs"]["joints_angles"] #n
                    elif "absolute" in arg_dict["robot_action"]:
                        action = info['o']["actual_state"]
                    else:
                        action = [0,0,0]

            if arg_dict["control"] == "oraculum":
                if t == 0:
                    action = env.action_space.sample()
                else:

                    if "absolute" in arg_dict["robot_action"]:
                        action = info['o']["goal_state"]
                    else:
                        print("ERROR - Oraculum mode only works for absolute actions")
                        quit()


            elif arg_dict["control"] == "keyboard":
                keypress = p.getKeyboardEvents()
                #print(action)
                action =  detect_key(keypress,arg_dict,action)
            elif arg_dict["control"] == "random":
                action = env.action_space.sample()

            observation, reward, done, truncated, info = env.step(action)

            if arg_dict["vtrajectory"] == True:
                visualize_trajectories(info,action)
            if arg_dict["vinfo"] == True:
                visualize_infotext(action, env, info)
            if "step" in arg_dict["robot_action"]:
                action[:3] = [0,0,0]

            if arg_dict["visualize"]:
                visualizations = [[],[]]
                env.render("human")
                for camera_id in range(len(env.cameras)):
                    camera_render = env.render(mode="rgb_array", camera_id=camera_id)
                    image = cv2.cvtColor(camera_render[camera_id]["image"], cv2.COLOR_RGB2BGR)
                    depth = camera_render[camera_id]["depth"]
                    image = cv2.copyMakeBorder(image, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(image, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[0].append(image)
                    depth = cv2.copyMakeBorder(depth, 30, 10, 10, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    cv2.putText(depth, 'Camera {}'.format(camera_id), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5,
                                (0, 0, 0), 1, 0)
                    visualizations[1].append(depth)

                if len(visualizations[0])%2 !=0:
                        visualizations[0].append(255*np.ones(visualizations[0][0].shape, dtype=np.uint8))
                        visualizations[1].append(255*np.ones(visualizations[1][0].shape, dtype=np.float32))
                fig_rgb = np.vstack((np.hstack((visualizations[0][0::2])),np.hstack((visualizations[0][1::2]))))
                fig_depth = np.vstack((np.hstack((visualizations[1][0::2])),np.hstack((visualizations[1][1::2]))))
                cv2.imshow('Camera RGB renders', fig_rgb)
                cv2.imshow('Camera depthrenders', fig_depth)
                cv2.waitKey(1)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def test_model(env, model=None, implemented_combos=None, arg_dict=None, model_logdir=None, deterministic=False):

    try:
        if "multi" in arg_dict["algo"]:
            model_args = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][1]
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"], env=model_args[1].env)
        else:
            model = implemented_combos[arg_dict["algo"]][arg_dict["train_framework"]][0].load(arg_dict["model_path"])
    except:
        if (arg_dict["algo"] in implemented_combos.keys()) and (arg_dict["train_framework"] not in list(implemented_combos[arg_dict["algo"]].keys())):
            err = "{} is only implemented with {}".format(arg_dict["algo"],list(implemented_combos[arg_dict["algo"]].keys())[0])
        elif arg_dict["algo"] not in implemented_combos.keys():
            err = "{} algorithm is not implemented.".format(arg_dict["algo"])
        else:
            err = "invalid model_path argument"
        raise Exception(err)

    images = []  # Empty list for gif images
    success_episodes_num = 0
    distance_error_sum = 0
    vel= arg_dict["max_velocity"]
    force = arg_dict["max_force"]
    steps_sum = 0
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    #p.setRealTimeSimulation(1)
    #p.setTimeStep(0.01)

    for e in range(arg_dict["eval_episodes"]):
        done = False
        obs = env.reset()
        is_successful = 0
        distance_error = 0
        step_sum = 0
        while not done:
            steps_sum += 1
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            is_successful = not info['f']
            distance_error = info['d']
            if arg_dict["vinfo"] == True:
                visualize_infotext(action, env, info)


            if (arg_dict["record"] > 0) and (len(images) < 8000):
                render_info = env.render(mode="rgb_array", camera_id = arg_dict["camera"])
                image = render_info[arg_dict["camera"]]["image"]
                images.append(image)
                print(f"appending image: total size: {len(images)}]")

        success_episodes_num += is_successful
        distance_error_sum += distance_error

    mean_distance_error = distance_error_sum / arg_dict["eval_episodes"]
    mean_steps_num = steps_sum // arg_dict["eval_episodes"]

    print("#---------Evaluation-Summary---------#")
    print("{} of {} episodes ({} %) were successful".format(success_episodes_num, arg_dict["eval_episodes"], success_episodes_num / arg_dict["eval_episodes"]*100))
    print("Mean distance error is {:.2f}%".format(mean_distance_error * 100))
    print("Mean number of steps {}".format(mean_steps_num))
    print("#------------------------------------#")
    model_name = arg_dict["algo"] + '_' + str(arg_dict["steps"])
    file = open(os.path.join(model_logdir, "train_" + model_name + ".txt"), 'a')
    file.write("\n")
    file.write("#Evaluation results: \n")
    file.write("#{} of {} episodes were successful \n".format(success_episodes_num, arg_dict["eval_episodes"]))
    file.write("#Mean distance error is {:.2f}% \n".format(mean_distance_error * 100))
    file.write("#Mean number of steps {}\n".format(mean_steps_num))
    file.close()

    if arg_dict["record"] == 1:
        gif_path = os.path.join(model_logdir, "train_" + model_name + ".gif")
        imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=15)
        os.system('./utils/gifopt -O3 --lossy=5 -o {dest} {source}'.format(source=gif_path, dest=gif_path))
        print("Record saved to " + gif_path)
    elif arg_dict["record"] == 2:
        video_path = os.path.join(model_logdir, "train_" + model_name + ".avi")
        height, width, layers = image.shape
        out = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))
        for img in images:
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.release()
        print("Record saved to " + video_path)


def main():
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="random", help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectgory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")
    arg_dict = get_arguments(parser)
    model_logdir = os.path.dirname(arg_dict.get("model_path", ""))
    # Check if we chose one of the existing engines
    arg_dict["engine"] = "pybullet"
    arg_dict["vinfo"] = True

    print("Path to the model using --model_path argument not specified. Testing random actions in selected environment.")
    arg_dict["gui"] = 1
    env_arguments = {
        "render_on": True, "visualize": arg_dict["visualize"], "workspace": arg_dict["workspace"],
                    "robot": arg_dict["robot"], "robot_init_joint_poses": arg_dict["robot_init"],
                    "robot_action": arg_dict["robot_action"],"max_velocity": arg_dict["max_velocity"],
                    "max_force": arg_dict["max_force"],"task_type": arg_dict["task_type"],
                    "action_repeat": arg_dict["action_repeat"],
                    "task_objects":arg_dict["task_objects"], "observation":arg_dict["observation"], "distractors":arg_dict["distractors"],
                    "num_networks":arg_dict.get("num_networks", 1), "network_switcher":arg_dict.get("network_switcher", "gt"),
                    "distance_type": arg_dict["distance_type"], "used_objects": arg_dict["used_objects"],
                    "active_cameras": arg_dict["camera"], "color_dict":arg_dict.get("color_dict", {}),
                    "max_steps": arg_dict["max_episode_steps"], "visgym":arg_dict["visgym"],
                    "reward": arg_dict["reward"], "logdir": arg_dict["logdir"], "vae_path": arg_dict["vae_path"],
                    "yolact_path": arg_dict["yolact_path"], "yolact_config": arg_dict["yolact_config"],
                    "natural_language": bool(arg_dict["natural_language"]),
                    "training": False, "publish_to_ros": True
                    }
    env_arguments["gui_on"] = arg_dict["gui"]

    env = gym.make(arg_dict["env_name"], **env_arguments)

    test_env(env, arg_dict)


if __name__ == "__main__":
    main()
