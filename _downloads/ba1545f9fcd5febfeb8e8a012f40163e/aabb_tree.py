"""
AABB Trees
==========

Building an axis-aligned bounding box (AABB) for a given trimesh
"""

import numpy as np
import pyvista as pv

import mirage as mr
import mirage.vis as mrv


def ray_triangle_intersection(ray_origin, ray_direction, vertices):
    vertex0 = vertices[0, :]
    vertex1 = vertices[1, :]
    vertex2 = vertices[2, :]
    EPSILON = 1e-6
    edge1, edge2 = vertex1 - vertex0, vertex2 - vertex0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return None  # This means the ray is parallel to this triangle.

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > EPSILON:  # ray intersection
        intersection_point = ray_origin + ray_direction * t
        return t, u, v, intersection_point
    else:  # This means that there is a line intersection but not a ray intersection.
        return None


def ray_box_intersection(ray_origin, ray_direction, box_lims):
    box_min = np.array([box_lims[0], box_lims[2], box_lims[4]])
    box_max = np.array([box_lims[1], box_lims[3], box_lims[5]])
    t_min = np.zeros(3, dtype=np.float32)
    t_max = np.zeros(3, dtype=np.float32)

    for i in range(3):  # Process each dimension independently
        if ray_direction[i] != 0:
            t_min[i] = (box_min[i] - ray_origin[i]) / ray_direction[i]
            t_max[i] = (box_max[i] - ray_origin[i]) / ray_direction[i]
        else:
            t_min[i] = float('-inf') if ray_origin[i] < box_min[i] else float('inf')
            t_max[i] = float('inf') if ray_origin[i] > box_max[i] else float('-inf')

        if t_min[i] > t_max[i]:
            t_min[i], t_max[i] = t_max[i], t_min[i]

    final_t_min = max(t_min)
    final_t_max = min(t_max)

    if final_t_min > final_t_max or final_t_max < 0:
        return None, None  # No intersection

    # Calculate intersection points
    intersection_min = ray_origin + final_t_min * ray_direction

    return intersection_min, final_t_min


def points_in_box(points, bounds) -> np.ndarray[bool]:
    check_pts = points[:, 0] > bounds[0]
    check_pts &= points[:, 0] < bounds[1]
    check_pts &= points[:, 1] > bounds[2]
    check_pts &= points[:, 1] < bounds[3]
    check_pts &= points[:, 2] > bounds[4]
    check_pts &= points[:, 2] < bounds[5]
    ret = np.argwhere(check_pts).flatten()
    return ret


def build_boxes(
    all_points: np.ndarray, split_verts: np.ndarray[int], max_depth: int, depth: int = 0
):
    if depth == max_depth:
        return
    boxes = {}
    if split_verts.size == 0:
        return
    mins = np.min(all_points[split_verts, :], axis=0)
    maxs = np.max(all_points[split_verts, :], axis=0)
    split_axis = np.argmax(maxs - mins)
    mid_pt = mins[split_axis] + (maxs[split_axis] - mins[split_axis]) / 2
    if split_axis == 0:
        boxes['left_bounds'] = (mins[0], mid_pt, mins[1], maxs[1], mins[2], maxs[2])
        boxes['right_bounds'] = (mid_pt, maxs[0], mins[1], maxs[1], mins[2], maxs[2])
    if split_axis == 1:
        boxes['left_bounds'] = (mins[0], maxs[0], mins[1], mid_pt, mins[2], maxs[2])
        boxes['right_bounds'] = (mins[0], maxs[0], mid_pt, maxs[1], mins[2], maxs[2])
    if split_axis == 2:
        boxes['left_bounds'] = (mins[0], maxs[0], mins[1], maxs[1], mins[2], mid_pt)
        boxes['right_bounds'] = (mins[0], maxs[0], mins[1], maxs[1], mid_pt, maxs[2])

    boxes['left_members'] = points_in_box(all_points, boxes['left_bounds'])
    boxes['right_members'] = points_in_box(all_points, boxes['right_bounds'])
    boxes['depth'] = depth

    if depth + 1 == max_depth:
        return boxes

    boxes['left_leaves'] = build_boxes(
        all_points, boxes['left_members'], max_depth=max_depth, depth=depth + 1
    )
    boxes['right_leaves'] = build_boxes(
        all_points, boxes['right_members'], max_depth=max_depth, depth=depth + 1
    )
    if boxes['left_leaves'] is None:
        del boxes['left_leaves']
    if boxes['right_leaves'] is None:
        del boxes['right_leaves']
    return boxes


def aggregate_children(pl, boxes, block):
    block.extend(
        [pv.Box(bounds=boxes['left_bounds']), pv.Box(bounds=boxes['right_bounds'])]
    )
    if 'left_leaves' in boxes:
        aggregate_children(pl, boxes['left_leaves'], block)
    if 'right_leaves' in boxes:
        aggregate_children(pl, boxes['right_leaves'], block)


def trace_indices(ray_origin, ray_direction, face_indices, faces_vertices):
    f, v = faces_vertices
    good_res = []
    for fi in face_indices:
        fiv = v[f[fi]]
        res = ray_triangle_intersection(ray_origin, ray_direction, fiv)
        if res is not None:
            good_res.append(res)
    return good_res


def trace_children(pl, ray_origin, ray_direction, boxes, faces_vertices):
    lpt, ltime = ray_box_intersection(ray_origin, ray_direction, boxes['left_bounds'])
    rpt, rtime = ray_box_intersection(ray_origin, ray_direction, boxes['right_bounds'])
    go_left = None

    if lpt is not None:  # and rpt is None (implied)
        go_left = True
    if rpt is not None:  # and lpt is None (implied)
        go_left = False
    if lpt is not None and rpt is not None:
        if ltime < rtime:
            go_left = True
        if rtime < ltime:
            go_left = False

    if go_left:
        if 'left_leaves' in boxes:
            pl.add_mesh(
                pv.Box(bounds=boxes['left_bounds']),
                line_width=10,
                style='wireframe',
                color='k',
            )
            return trace_children(
                pl, ray_origin, ray_direction, boxes['left_leaves'], faces_vertices
            )
        else:
            return trace_indices(
                ray_origin, ray_direction, boxes['left_members'], faces_vertices
            )
    else:
        if 'right_leaves' in boxes:
            pl.add_mesh(
                pv.Box(bounds=boxes['right_bounds']),
                line_width=10,
                style='wireframe',
                color='k',
            )
            return trace_children(
                pl, ray_origin, ray_direction, boxes['right_leaves'], faces_vertices
            )
        else:
            return trace_indices(
                ray_origin, ray_direction, boxes['right_members'], faces_vertices
            )


obj = mr.SpaceObject('stanford_dragon.obj')


obj.v = obj.v.astype(np.float32)
mr.tic('Building AABB')
boxes = build_boxes(
    obj.face_centroids, np.arange(obj.face_centroids.shape[0]), max_depth=11
)
mr.toc()


pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, opacity=1.0)
block = pv.MultiBlock()
aggregate_children(pl, boxes, block)
pl.add_mesh(block, style='wireframe', color='r')
pl.camera.position = (3.5, 0.0, 0.0)
pl.show()

# %%
# Now that we have some boxes, let's see how much faster they make our raytracing
ray_origin = 10 * np.array([0.1, 1.0, -0.1])
ray_direction = -mr.hat(ray_origin)

mr.tic('Brute force tracing')
good_res = []
for f in obj.f:
    x = ray_triangle_intersection(ray_origin, ray_direction, obj.v[f])
    if x is not None:
        good_res.append(x)
mr.toc()

pl = pv.Plotter()
mrv.render_spaceobject(pl, obj, opacity=1.0)

for x in good_res:
    mrv.scatter3(pl, x[-1], point_size=40, color='m')


mr.tic('AABB tracing')
res = trace_children(pl, ray_origin, ray_direction, boxes, (obj.f, obj.v))
mr.toc()

for x in res:
    mrv.scatter3(pl, x[-1], point_size=100, color='lime', opacity=0.2)

pl.camera.position = (3.5, 0.0, 0.0)
pl.show()
