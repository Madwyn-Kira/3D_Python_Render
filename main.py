from matplotlib import pyplot as plt, animation
from matplotlib.animation import PillowWriter
from core import vertexes_to_projective
from parse_file import file_parse
from convert import array_converter
from vizualize import print_plot
from affine_transformations import rotation_matrix, transfer_matrix, tensile_matrix
from camera import lookAt
from viewsss import view_projections
from illumination import lambert, phong
from PIL import Image
import numpy as np


if __name__ == "__main__":
    def rotation(ang):
        rot = np.array(
            [[np.cos(ang), 0, np.sin(ang), 0],
             [0, 1, 0, 0],
             [-np.sin(ang), 0, np.cos(ang), 0],
             [0, 0, 0, 1]],
            dtype=np.float64
        )
        return rot
    file_pth = 'african_head.obj'
    result_array = array_converter.sz_conv(
        file_parse.parse_vector(file_pth, 'v'), 800)
    f_array = file_parse.parse_place(file_pth, 'v')
    texture_v = file_parse.parse_vector(file_pth,
                                        'vt')
    texture_faces = file_parse.parse_place(
        file_pth, 'vt')
    normal_v = file_parse.parse_vector(file_pth,
                                       'vn')
    normal_faces = file_parse.parse_place(
        file_pth, 'vn')
    image = Image.open('african_head_diffuse.tga')
    imag = np.zeros((800, 800, 3), dtype=np.uint8)
    cam_point = np.array([-60, 10, 750], dtype=np.float64)
    frames = []
    fig = plt.figure()
    N = 100
    params = [[0, 0, 0], [1, 1, 1], [200, 0, 0]]
    result_array = rotation_matrix.magic_func(result_array, params, 'y')
    for i in range(N):
        ang = np.pi / N
        # print(ang)
        params = [[50, ang, 50], [1, 1, 1], [20, 0, 0]]
        # params = [[20, 20, 20], [0, 0, 0], [0, 0, 0]]
        result_array = rotation_matrix.magic_func(result_array, params, 'y')
        # result_array = lookAt.cam_view_matrix(result_array,
        #                                        cam_point,
        #                                       np.array([0,
        #                                                 0,
        #                                                 0], dtype=np.float64))

        # cam_point[0] = -60 + 70 * np.cos(i)
        # cam_point[1] = 10 + 70 * np.cos(i)

        # lambert.lambert_illumination(result_array, f_array, np.array([-60, 10, 650], dtype=np.float64), True, True)
        # result_array = view_projections.orthogonal_projection(result_array)
        # result_array = view_projections.viewport(result_array, 512, 512)
        # phong.phong_illumination(result_array, cam_point,
        #                          f_array, np.array([-100, 80, 650], dtype=np.float64), texture_v, normal_v, normal_faces, texture_faces, image, imag)

        # np.array([-60, 10, 650], dtype=np.float64)
        gg = lambert.lambert_illumination(result_array, f_array, np.array([-60, 10, 1650], dtype=np.float64), True, True, texture_v, texture_faces, image, imag)

        # print_plot.print_image(result_array, f_array, True)
        im = plt.imshow(gg)
        frames.append([im])

# gif animation creation
ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
writer = PillowWriter(fps=60)
plt.show()
ani.save("line.gif", writer=writer)