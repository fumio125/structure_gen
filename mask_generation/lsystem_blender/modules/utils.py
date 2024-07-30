import math
import bmesh
import bpy
import colorsys
import numpy as np
import random
import os
import sys


def make_leaf_colormap(obj, h):
    if not obj.data.vertex_colors:
        obj.select_set(state=True)
        bpy.ops.mesh.vertex_color_add()
    vc = obj.data.vertex_colors[0].data

    for i in range(len(vc)):
        vc[i].color = colorsys.hsv_to_rgb(h, 1, 1) + (1,)

    return True


# 頂点カラー表示のマテリアルを作成
def make_material_vertexcolor(obj, material_name="VertexColorMaterial", arg_addnodename="ShaderNodeAttribute",
                              arg_jointnodename="Principled BSDF"):
    if obj == None:
        return False
    if obj.type != 'MESH':
        return False

    for material_slot in obj.material_slots:
        obj.active_material = material_slot.material
        bpy.ops.object.material_slot_remove()

    old_mesh = obj.data
    new_mesh = bpy.data.meshes.new(name="NewMesh")
    bm = bmesh.new()
    bm.from_mesh(old_mesh)
    bm.to_mesh(new_mesh)
    bm.free()

    vertex_material = bpy.data.materials.new("VertexColor")
    vertex_material.use_nodes = True

    bpy.ops.object.material_slot_add()
    obj.active_material = vertex_material

    mat_nodes = vertex_material.node_tree.nodes
    attr_node = mat_nodes.new(type="ShaderNodeVertexColor")
    # attr_node.name = arg_addnodename
    # attr_node.attribute_name = "Col"

    mat_link = vertex_material.node_tree.links
    joint_node = mat_nodes[arg_jointnodename]
    mat_link.new(attr_node.outputs[0], joint_node.inputs[0])

    return True


def make_obj_material_based_rgb(obj, rgb):
    old_mesh = obj.data
    new_mesh = bpy.data.meshes.new(name="NewMesh")
    bm = bmesh.new()
    bm.from_mesh(old_mesh)
    bm.to_mesh(new_mesh)
    bm.free()
    material = bpy.data.materials.new(name="NewMaterial")
    material.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
    material.specular_intensity = 0
    material.roughness = 0
    new_mesh.materials.append(material)
    obj.data = new_mesh
    obj.show_name = False
    return True


def render_plant(plant_idx, cam_name, res, photo_num, radius, img_path, dic, parent_pos, obj_name, dtype):
    scene = bpy.context.scene
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.image_settings.file_format = str('PNG')
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = False  # 背景透過の有無
    scene.render.use_persistent_data = True  # レンダリングデータを保存して再レンダリングを高速化するか
    scene.display.shading.type = 'SOLID'
    # bpy.types.View3Dshading = 'RANDOM'

    # Worldオブジェクトを取得または作成
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True

    # 背景を白色に設定
    # world.use_nodes = True

    # workbench = scene.render.workbench
    print(bpy.context.scene.render)

    cam = scene.objects[cam_name]
    camera_parent = bpy.data.objects.new("CameraParent", None)
    if dtype == 'plant':
        camera_parent.location = (0, 0, parent_pos)
    else:
        camera_parent.location = (0, 0, 0)
    scene.collection.objects.link(camera_parent)
    cam.parent = camera_parent

    camera_constraint = cam.constraints.new(type='TRACK_TO')
    camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    camera_constraint.up_axis = 'UP_Y'
    camera_constraint.target = camera_parent

    yaw_max = 2 * np.pi - (2 * np.pi / photo_num)
    yaw_angle = np.linspace(0, yaw_max, photo_num)

    for render_idx in range(photo_num):
        world.node_tree.nodes.clear()
        background_node = world.node_tree.nodes.new('ShaderNodeBackground')
        background_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        output_node = world.node_tree.nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

        if dtype == 'plant':
            cam_z = random.uniform(5, 10)
        elif dtype == 'amodal':
            cam_z = random.uniform(parent_pos, parent_pos + 5)
        else:
            cam_z = random.uniform(7, 9)
        cam.location = (0, radius, cam_z)

        yaw = yaw_angle[render_idx]
        camera_parent.rotation_euler = np.array([0, 0, yaw])

        # rendering
        h = 0
        intval = 1 / len(obj_name) if dtype == 'leaf' else 1 / (len(obj_name) - 1)
        for i, tgt_obj_name in enumerate(obj_name):
            obj = bpy.data.objects.get(tgt_obj_name)
            bpy.context.view_layer.objects.active = obj
            if dtype == 'leaf':
                make_leaf_colormap(obj, h)
                make_material_vertexcolor(obj)
                h = h + intval
            else:
                if i != 0:
                    make_leaf_colormap(obj, h)
                    make_material_vertexcolor(obj)
                    h = h + intval
                else:
                    vc = obj.data.vertex_colors[0].data
                    for i in range(len(vc)):
                        vc[i].color = colorsys.hsv_to_rgb(0.06, 0.42, 0.45) + (1,)
                    make_material_vertexcolor(obj)

        filename = f'plant_{str(plant_idx).zfill(7)}_{str(render_idx).zfill(4)}'
        scene.render.filepath = os.path.join(img_path, 'img', filename)
        bpy.ops.render.render(write_still=True)
        if dtype == 'amodal':
            render_amodal_mask(tgt_obj_names=obj_name, pid=plant_idx, num_idx=render_idx, scene=scene,
                               image_path=img_path)
        for tgt_obj_name in obj_name:
            obj = bpy.data.objects.get(tgt_obj_name)
            obj.hide_render = False
            make_obj_material_based_rgb(obj, (1.0, 1.0, 1.0))
        render_per_organ(tgt_obj_names=obj_name, pid=plant_idx, num_idx=render_idx, scene=scene, image_path=img_path,
                         dtype=dtype)
        dic['camera_position'].append((radius * math.cos(yaw), radius * math.sin(yaw), cam_z))

    bpy.data.objects.remove(camera_parent, do_unlink=True)
    return True


def render_per_organ(tgt_obj_names, pid, num_idx, scene, image_path, dtype):
    for obj_idx, obj_name in enumerate(tgt_obj_names):
        obj = bpy.data.objects.get(obj_name)
        bpy.context.view_layer.objects.active = obj
        make_obj_material_based_rgb(obj, (0.0, 0.0, 0.0))
        if dtype == 'leaf':
            filename = f'leaf_{str(obj_idx - 1).zfill(7)}'
        else:
            if obj_idx == 0:
                filename = f'branch_{str(obj_idx).zfill(7)}'
            else:
                filename = f'leaf_{str(obj_idx - 1).zfill(7)}'
        scene.render.filepath = os.path.join(image_path, 'mask', f'{str(pid).zfill(7)}',
                                             f'render_{str(num_idx).zfill(4)}', filename)
        bpy.ops.render.render(write_still=True)
        make_obj_material_based_rgb(obj, (1.0, 1.0, 1.0))


def render_amodal_mask(tgt_obj_names, pid, num_idx, scene, image_path):
    for obj_idx, obj_name in enumerate(tgt_obj_names):
        for obj in bpy.data.objects:
            if obj.name != obj_name:
                obj.hide_render = True
            else:
                obj.hide_render = False
        if obj_idx == 0:
            filename = f'branch_{str(obj_idx).zfill(7)}'
        else:
            filename = f'leaf_{str(obj_idx - 1).zfill(7)}'
        scene.render.filepath = os.path.join(image_path, 'amodal_seg', f'{str(pid).zfill(7)}',
                                             f'render_{str(num_idx).zfill(4)}', filename)
        bpy.ops.render.render(write_still=True)


def get_height_branch(obj):
    if obj.type == 'MESH' and obj.data is not None:
        mesh = obj.data
        vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
        min_z = min(vertices, key=lambda v: v.z).z
        max_z = max(vertices, key=lambda v: v.z).z
        height = max_z - min_z
        return height