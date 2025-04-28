import airsim
import time
import math

client = airsim.CarClient()
client.confirmConnection()

def get_yaw_from_quaternion(q):
    # Hàm chuyển quaternion sang góc Yaw (độ)
    siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    yaw_deg = math.degrees(yaw)
    return yaw_deg

while True:
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    orient = car_state.kinematics_estimated.orientation

    yaw_deg = get_yaw_from_quaternion(orient)

    print(f"Position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}, Yaw={yaw_deg:.2f}")
    time.sleep(0.5)