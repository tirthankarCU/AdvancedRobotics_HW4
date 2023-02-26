import numpy as np
import matplotlib.pyplot as plt
import cubic_spline_planner
from scipy.optimize import minimize

dt = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
max_steer = np.radians(30.0)  # [rad] max steering angle

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += (self.v / L ) * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

prev_err,integral_t,isFirstErr=0,0,True 
def pid_control(target, current,Kp,Ki,Kd):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """

    """
    GRAD TODO: Edit this function (and potentially others in this file) to implement the 
    integral and derivative components of the PID controller. They should take the form:
    Ki * integral_t=-H ^ t=0 (target_t - current_t) (use, e.g., the trapezoid rule for numerical integration)
    Kd * d_dx(target_t - current_t)|t=0 (calculate the numerical derivative)
    """
    global prev_err, isFirstErr, integral_t 
    err=(target - current)
    d_dx=0
    if isFirstErr:
        isFirstErr=False 
    else:
        integral_t+=0.5*dt*(prev_err+err)
        d_dx=(err-prev_err)/dt 
    prev_err=err
    return Kp * err + Ki * integral_t + Kd * d_dx

def p_control(target, current,Kp):
    err=(target - current)
    return Kp * err

def stanley_control(state, cx, cy, cyaw, last_target_idx,k):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx,error_front_axle


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    angle=angle%(2*np.pi)
    if angle > np.pi:
        angle -= 2.0 * np.pi

    if angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def main(coeff_arr,opt):
    global prev_err, isFirstErr, integral_t 
    prev_err,integral_t,isFirstErr=0,0,True 
    if opt==2:
        k,Kp,Ki,Kd=coeff_arr
    elif opt==1:
        k,Kp=coeff_arr
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    target_speed = 30.0 / 3.6  # [m/s]

    max_simulation_time = 100.0

    # Initial state
    state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)
    while max_simulation_time >= time and last_idx > target_idx:
        if opt==2:
            ai = pid_control(target_speed, state.v,Kp,Ki,Kd)
        elif opt==1:
            ai = p_control(target_speed, state.v,Kp)
        di, target_idx, error_front_axle = stanley_control(state, cx, cy, cyaw, target_idx,k)
        state.update(ai, di)

        time += dt
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"
    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal") 
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    opt=2
# OPTION 2
    if opt==2:  
        #[0.90880928,0.944044,0.013869,-0.817982]
        #[2.47291642e+00,2.83102609e+00,-5.63907686e-07,-5.27911749e-01]
        #[12.07378138,12.35622768,-0.17252543,-0.05639835]
        param_init=np.array([11.63813308,12.0552154,-0.11803027,-0.04411984]) #[0.5,1,0.1,0.1]
        main(param_init,2)
# OPTION 1
    elif opt==1:
        param_init=np.array([15.88447979,10.01846168]) #[0.5,1]
        main(param_init,1)