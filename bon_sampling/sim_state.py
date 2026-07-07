import mujoco
import numpy as np

STATE_SPEC = (
    mujoco.mjtState.mjSTATE_FULLPHYSICS
    | mujoco.mjtState.mjSTATE_CTRL
    | mujoco.mjtState.mjSTATE_MOCAP_POS
    | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    | mujoco.mjtState.mjSTATE_WARMSTART
)


def state_size(model):
    return mujoco.mj_stateSize(model, STATE_SPEC)


def get_sim_state(model, data):
    state = np.empty(state_size(model), dtype=np.float64)
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def set_sim_state(model, data, state):
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
