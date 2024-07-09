import numpy as np
import matplotlib.pyplot as plt
from dash import Input
from matplotlib.animation import FuncAnimation

# Define the vectors and matrices
p_world = np.array([0, 0, -5, 1])
q_world = np.array([1, 0, 0, 0])
scale = np.array([0.04, 0.01, 0.01])

W_camera_base = np.array([
    [1.0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, -1.12923026],
    [0, 0, 0, 1]
])

projection = np.array([
    [0.97427851, 0, 0, 0],
    [0, 1.73205066, 0, 0],
    [0, 0, -1.00250626, -0.100250624],
    [0, 0, -1, 0]
])


def get_rotated_W_camera(angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    R_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    W_camera_rotated = W_camera_base.copy()
    W_camera_rotated[:3, :3] = R_z @ W_camera_base[:3, :3]
    return W_camera_rotated


def gaussian_kernel(x, y, center, cov_inv):
    diff = np.array([x, y]) - center
    power = -0.5 * diff.T @ cov_inv @ diff
    return np.exp(power)


newPos = 0
ind = 0
def update(frame):
    global newPos, ind
    if ind == 0:
        ind = input()
    angle = np.radians(frame)
    W_camera = 0
    if frame < 150:
        W_camera = W_camera_base.copy()
        W_camera[0, 3] = -angle * 2
        newPos = W_camera[0, 3]
    else:
        W_camera = get_rotated_W_camera(angle - newPos/2)
        W_camera[0, 3] = newPos

    opacity = (360 - frame) / 360
    # Transformations
    p_view = W_camera @ p_world
    p_clip = projection @ p_view
    p_ndc = p_clip / p_clip[3]

    print(f"Frame: {frame}")

    screenPosX = ((p_ndc[0] + 1) * imageWidth - 1) * 0.5
    screenPosY = ((p_ndc[1] + 1) * imageHeight - 1) * 0.5

    # Covariance in 3D
    qw, qx, qy, qz = q_world
    R = np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])

    S = np.diag(scale)
    W_rotational = W_camera[:3, :3]
    cov3D = W_rotational @ S @ S.T @ W_rotational.T

    # Jacobian and covariance in 2D
    focalX = 1108.51233
    focalY = 623.538208
    t = p_view.copy()
    limx = 1.3 * 1.02640057
    limy = 1.3 * 0.577350318
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    J = np.array([
        [focalY / t[2], 0., -focalY * t[0] / t[2] ** 2],
        [0., focalY / t[2], -focalY * t[1] / t[2] ** 2],
        [0., 0., 0.]
    ])

    cov2D = J @ W_rotational @ cov3D @ W_rotational.T @ J.T
    cov2D = cov2D[:2, :2]
    cov2D_inv = np.linalg.inv(cov2D)

    # Generate the image
    Z = np.vectorize(lambda x, y: gaussian_kernel(x, y, (screenPosX, screenPosY), cov2D_inv))(X, Y)

    ax.clear()
    ax.imshow(Z, extent=(0, imageWidth, 0, imageHeight), origin='lower', cmap='hot', vmin=0, vmax=1)
    ax.set_title(f"Gaussian Distribution\ntime (t): {frame}")


# Animation setup
imageWidth = 50
imageHeight = 50

x = np.linspace(0, imageWidth, imageWidth)
y = np.linspace(0, imageHeight, imageHeight)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()

ani = FuncAnimation(fig, update, frames=range(0, 360, 10), repeat=True)
plt.show()
