import numpy as np
import open3d as o3d
from sympy import false


def read_from_WADS(data_path="", label_path="",
                   duplicated_removal=True  # follows 4DenoiseNet
                   ):
    data = np.fromfile(data_path, np.float32).reshape([-1, 4])
    label = np.fromfile(label_path, np.uint32).reshape([-1])
    for i in range(len(label)):
        label[i] = 1 if label[i] == 110 else 0  # in WADS, the value "110" represents "active falling snow"

    if duplicated_removal:
        data, idx_unique = np.unique(data, axis=0, return_index=True)  # duplicated points removal
        label = label[idx_unique]  # duplicated points removal

    return data, label


def analyse_and_visualize(pc_data, pc_label, snows: dict = None, visualize_it=False):
    tp, tn, fp, fn = 0, 0, 0, 0  # true positive, true negative, false positive, false negative
    color = np.zeros([len(pc_label), 3])

    for j in range(len(pc_label)):
        if pc_label[j] == 1:
            if j in snows.keys():
                tp += 1
                color[j] = [0, 154 / 255., 85 / 255.]  # green
            else:
                fn += 1
                color[j] = [182 / 255., 48 / 255., 28 / 255.]  # red
        else:
            if j in snows.keys():
                fp += 1
                color[j] = [0, 110 / 255., 184 / 255.]  # blue
            else:
                tn += 1  # black, which is default in open3d

    analysis = [('true positive', tp), ('true negative', tn),
                ('false positive', fp), ('false negative', fn),
                ('precision', 0 if tp + fp == 0 else 1.0 * tp / (tp + fp)),
                ('recall', 0 if tp + fn == 0 else 1.0 * tp / (tp + fn))]
    print(analysis)

    if not visualize_it:
        return analysis

    # original
    pc_o3d_original = o3d.geometry.PointCloud()
    pc_o3d_original.points = o3d.utility.Vector3dVector(pc_data[:, :3])
    original_color = np.zeros([len(pc_data), 3])
    pc_o3d_original.colors = o3d.utility.Vector3dVector(original_color)

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='original', width=600, height=600, left=0, top=20)
    vis1.add_geometry(pc_o3d_original)

    # colored
    pc_o3d_colored = o3d.geometry.PointCloud()
    pc_o3d_colored.points = o3d.utility.Vector3dVector(pc_data[:, :3])
    pc_o3d_colored.colors = o3d.utility.Vector3dVector(color)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='colored', width=600, height=600, left=610, top=20)
    vis2.add_geometry(pc_o3d_colored)

    # de-snowed
    mask = np.zeros([len(pc_data)], dtype=np.bool)
    for i in range(len(pc_data)):
        if i not in snows:
            mask[i] = True
    pc_data = pc_data[mask]
    original_color = original_color[mask]
    pc_o3d_desnowed = o3d.geometry.PointCloud()
    pc_o3d_desnowed.points = o3d.utility.Vector3dVector(pc_data[:, :3])
    pc_o3d_desnowed.colors = o3d.utility.Vector3dVector(original_color)

    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name='desnowed', width=600, height=600, left=1220, top=20)
    vis3.add_geometry(pc_o3d_desnowed)

    while True:
        vis1.update_geometry(pc_o3d_original)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        vis2.update_geometry(pc_o3d_colored)
        if not vis2.poll_events():
            break
        vis2.update_renderer()
        vis3.update_geometry(pc_o3d_desnowed)
        if not vis3.poll_events():
            break
        vis3.update_renderer()

    vis1.destroy_window()
    vis2.destroy_window()
    vis3.destroy_window()

    return analysis


# open3d viewing angle, use Ctrl+C and Ctrl+V to view
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" :
	[
		{
			"boundingbox_max" : [ 183.19197082519531, 97.743217468261719, 24.21990966796875 ],
			"boundingbox_min" : [ -199.65153503417969, -101.93829345703125, -3.3196868896484375 ],
			"field_of_view" : 60.0,
			"front" : [ 0.04240589830333176, -0.71521369442337035, 0.69761816998868476 ],
			"lookat" : [ -1.2195525796371856, -0.31706689589310921, 0.88953746950206103 ],
			"up" : [ -0.0076877807456522881, 0.6979920104866868, 0.71606427876550294 ],
			"zoom" : 0.040000000000000001
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}