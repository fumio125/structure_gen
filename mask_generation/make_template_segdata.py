import math
import bmesh
import bpy
import colorsys
import exec
import numpy as np
import random
import json
import os
import sys
import time

import builtins
import contextlib

from utils import *


@contextlib.contextmanager
def suppress_print():
    orig_print = builtins.print
    builtins.print = lambda *args, **kwargs: None
    try:
        yield
    finally:
        builtins.print = orig_print


# create a string which models a leaf
def create_leaf_string(dtype):
    if dtype == 'ara':
        L_list = []
        la_list = ["0.0008", "0.002"]
        ls_list = [[0.0032, 0.004], [0.008, 0.01]]
        lb_list = ["0.0008", "0.002"]
        ll_list = ["0.0096", "0.024"]

        for idx in range(len(la_list)):
            leaf = exec.Exec()
            leaf.define("LA", la_list[idx])
            leaf.define("RA", "1.0")
            leaf.define("LS", str(random.uniform(ls_list[idx][0], ls_list[idx][1])))
            leaf.define("RS", "1.2")
            leaf.define("LB", lb_list[idx])
            leaf.define("RB", "1.1")
            leaf.define("LL", ll_list[idx])
            leaf.define("RL", "1.05")
            leaf.define("DB", "0.25")
            leaf.define("DL", "0.8")

            leaf.set_axiom("p(surface)F(0)A(6)")
            leaf.add_rule("A(t)", "f(0)[-B(8)F(0)][C(add(t,1))][+B(8)F(0)]")
            leaf.add_rule("B(t)", "f(LB,RB)B(sub(t,DB))", condition="gt(t,0)")
            leaf.add_rule("C(t)", "f(LA,RA)[-B(t)F(0)][C(add(t,1))][+B(t)F(0)]", condition="lt(t,7)")
            leaf.add_rule("C(t)", "f(LS,RS)[-D(t)F(0)][C(add(t,1))][+D(t)F(0)]", condition="gteq(t,7)")
            leaf.add_rule("D(t)", "f(LL,RL)D(sub(t,DL))", condition="gt(t,0)")
            leaf.add_rule("f(s,r)", "f(mul(s,r),r)")

            leaf.exec(min_iterations=20, angle=70)
            L_list.append(leaf.turtle_string)
            bpy.ops.object.delete(use_global=False, confirm=False)

        return L_list
    elif dtype == 'komatsuna':
        L_list = []
        la_list = ["0.002", "0.02", "0.03"]
        ls_list = ["0.003", "0.01", "0.015"]
        lb_list = ["0.0008", "0.01", "0.015"]
        ll_list = ["0.024", "0.032", "0.048"]

        for idx in range(len(la_list)):
            leaf = exec.Exec()
            leaf.define("LA", la_list[idx])
            leaf.define("RA", "1.05")
            leaf.define("LS", ls_list[idx])
            leaf.define("RS", "1.2")
            leaf.define("LB", lb_list[idx])
            leaf.define("RB", "1.1")
            leaf.define("LL", ll_list[idx])
            leaf.define("RL", "1.05")
            leaf.define("DB", "0.50")
            leaf.define("DL", "0.80")

            if idx == 0:
                leaf.set_axiom("p(surface)F(0)A(6)")
                leaf.add_rule("A(t)", "f(0)[-(70)B(8)F(0)][C(add(t,1))][+(70)B(8)F(0)]")
                leaf.add_rule("B(t)", "f(LB,RB)B(sub(t,DB))", condition="gt(t,0)")
                leaf.add_rule("C(t)", "f(LA,RA)[-(48)B(t)F(0)][C(add(t,1))][+(48)B(t)F(0)]", condition="lt(t,7)")
                leaf.add_rule("C(t)", "f(LS,RS)[-(48)D(t)F(0)][C(add(t,1))][+(48)D(t)F(0)]", condition="gteq(t,7)")
                leaf.add_rule("D(t)", "f(LL,RL)D(sub(t,DL))", condition="gt(t,0)")
                leaf.add_rule("f(s,r)", "f(mul(s,r),r)")
                leaf.exec(min_iterations=20)
            else:
                leaf.set_axiom("p(surface)F(0)A(9)")
                leaf.add_rule("A(t)", "f(0)[-B(8)F(0)][C(add(t,1))][+B(8)F(0)]")
                leaf.add_rule("B(t)", "f(LB,RB)B(sub(t,DB))", condition="gt(t,0)")
                leaf.add_rule("C(t)", "f(LA,RA)[-B(t)F(0)][C(add(t,1))][+B(t)F(0)]", condition="lt(t,10)")
                leaf.add_rule("C(t)", "f(LS,RS)[-D(t)F(0)][C(add(t,1))][+D(t)F(0)]", condition="gteq(t,10)")
                leaf.add_rule("D(t)", "f(LL,RL)D(sub(t,DL))", condition="gt(t,0)")
                leaf.add_rule("f(s,r)", "f(mul(s,r),r)")
                leaf.exec(min_iterations=20, angle=75)
            L_list.append(leaf.turtle_string)
            bpy.ops.object.delete(use_global=False, confirm=False)

        return L_list
    elif dtype == 'amodal':
        leaf_size = 0.30
        len_peti = 0.50
        leaf_angle = 7.0

        # Future works: string must be made from leaf l-system above
        L = f"f({len_peti})p(surface)[[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})A{{F(0.0)]F(0.0)CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}][[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})B{{F(0.0)]F(0.0)CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}]"
    else:
        leaf_size = 0.40
        len_peti = 0.70
        leaf_angle = 7.0

        # Future works: string must be made from leaf l-system above
        L = f"p(skin)¤(0.02)F(0.01)F({len_peti})p(surface)[[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})[+({leaf_angle})A{{F(0.0)]F(0.0)CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}][[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})[-({leaf_angle})B{{F(0.0)]F(0.0)CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}{{F(0.0)]F(0.0)f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})f({leaf_size})CF(0.0)}}]"

    return L


# create L-system by DATA_TYPE
def create_lsystem(dtype, L):
    lsys = exec.Exec()
    if dtype == 'ara':
        leaf_str_list = L["string"]
        peti1_width = f"¤(rand(0.03, 0.04))"
        peti2_width = f"¤(rand(0.05, 0.07))"
        peti_length_list = [[0.0, 0.5], [0.6, 1.0]]
        leaf1_num = L["leaf1_num"]
        leaf2_num = L["leaf2_num"]
        angle1_list = L["angle1_list"]
        angle2_list = L["angle2_list"]
        sa1 = f"^(rand(40, 50))"
        sa2 = f"^(rand(60, 70))"

        start_idx = random.randint(1, 2)
        if start_idx == 1:
            l = f"[{sa1}{peti1_width}f(0.0000000001)F(rand({peti_length_list[0][0]}, {peti_length_list[0][1]})){leaf_str_list[0]}]"
            for i in range(1, leaf1_num):
                l += f"[:/({angle1_list[i]}){sa1}{peti1_width}f(0.0000000001)F(rand({peti_length_list[0][0]}, {peti_length_list[0][1]})){leaf_str_list[0]};]"
            for j in range(leaf2_num):
                l += f"[:/({angle2_list[j]}){sa2}{peti2_width}f(0.0000000001)F(rand({peti_length_list[1][0]}, {peti_length_list[1][1]})){leaf_str_list[1]};]"
        else:
            l = f"[{sa2}{peti2_width}f(0.0000000001)F(rand({peti_length_list[1][0]}, {peti_length_list[1][1]})){leaf_str_list[1]}]"
            for i in range(leaf1_num):
                l += f"[:/({angle1_list[i]}){sa1}{peti1_width}f(0.0000000001)F(rand({peti_length_list[0][0]}, {peti_length_list[0][1]})){leaf_str_list[0]};]"
            for j in range(1, leaf2_num):
                l += f"[:/({angle2_list[j]}){sa2}{peti2_width}f(0.0000000001)F(rand({peti_length_list[1][0]}, {peti_length_list[1][1]})){leaf_str_list[1]};]"

        lsys.set_axiom("X")
        lsys.add_rule("X", f"{l}Y")
    elif dtype == 'komatsuna':
        leaf_str_list = L["string"]
        ipeti_width = f"¤(rand(0.02, 0.03))"
        speti_width = f"¤(rand(0.03, 0.04))"
        bpeti_width = f"¤(rand(0.06, 0.07))"
        peti_length_list = [[0.2, 0.25], [0.3, 0.4], [0.6, 0.8]]
        sleaf_num = L["sleaf_num"]
        bleaf_num = L["bleaf_num"]
        sangle_list = L["sangle_list"]
        bangle_list = L["bangle_list"]
        isa = f"^(rand(75, 80))"
        ssa = f"^(rand(50, 60))"
        bsa = f"^(rand(65, 70))"
        l = f"[{isa}{ipeti_width}f(0.0000000001)F(rand({peti_length_list[0][0]}, {peti_length_list[0][1]})){leaf_str_list[0]}]"
        l += f"[:/(rand(170, 190)){isa}{ipeti_width}f(0.0000000001)F(rand({peti_length_list[0][0]}, {peti_length_list[0][1]})){leaf_str_list[0]};]"
        if sleaf_num > 0:
            for i in range(sleaf_num):
                l += f"[:/({sangle_list[i]}){ssa}{speti_width}f(0.0000000001)F(rand({peti_length_list[1][0]}, {peti_length_list[1][1]})){leaf_str_list[1]};]"
        if bleaf_num > 0:
            for i in range(bleaf_num):
                l += f"[:/({bangle_list[i]}){bsa}{bpeti_width}f(0.0000000001)F(rand({peti_length_list[2][0]}, {peti_length_list[2][1]})){leaf_str_list[2]};]"

        lsys.set_axiom("X")
        lsys.add_rule("X", f"{l}Y")
    elif dtype == 'amodal':
        # environment value
        a = (17, 40)
        b = (30, 45)
        l = (1, 2)
        d1 = (80, 100)
        d2 = (120, 150)
        t1 = (70, 100)
        t2 = (70, 90)
        t3 = (80, 110)

        # set modeling string
        sa = f"&(rand({a[0]}, {a[1]}))"
        sb = f"&(rand({b[0]}, {b[1]}))"
        sf = f"F(rand({l[0]}, {l[1]}))"
        sd1 = f"/(rand({d1[0]}, {d1[1]}))"
        sd2 = f"/(rand({d2[0]}, {d2[1]}))"
        st1 = f"/(rand({t1[0]}, {t1[1]}))"
        st2 = f"/(rand({t2[0]}, {t2[1]}))"
        st3 = f"/(rand({t3[0]}, {t3[1]}))"
        # 互生
        b_m = f"{sf}[:^(65){L};]{sf}[:&(65){L};]{sf}"
        # 対生
        b_o = f"{sf}[:^(65){L};][:&(65){L};]{sf}"
        # 十字対生
        b_co = f"{sf}[:^(65){L};][:&(65){L};]{sf}[:/(90)^(65){L};][:/(90)&(65){L};]{sf}"
        # 輪生
        b_r = f"{sf}[:^(65){L};][:&(65){L};][:/(90)^(65){L};][:/(90)&(65){L};]{sf}"

        lsys.delete()

        # define lsystem
        lsys.define("lr", "1.109")
        lsys.define("vr", "1.732")

        lsys.set_axiom(f"p(skin)¤(0.1)F(0.0001)F(1)/(45)X")
        lsys.add_rule("X", f"¤(mul(0.1,vr))F(4)[{sa}Y]{sd1}[{sa}Y]{sd2}[{sa}Y]")
        # lsys.add_rule("X", f"¤(mul(0.1,vr))F(4)[{sa}{b_o}Y]{st1}[{sa}{b_o}Y]{st2}[{sa}{b_o}Y]{st3}[{sa}{b_o}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4)[{sa}{b_o}Y]{sd1}{sd2}[{sa}{b_o}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4){sd1}[{sa}{b_o}Y]{sd2}[{sa}{b_o}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4)[{sa}{b_o}Y]{sd1}[{sa}{b_o}Y]{sd2}")
        lsys.add_rule("F(l)", "F(mul(l,lr))")
        lsys.add_rule("¤(w)", "¤(mul(w,vr))")
    else:
        # environment value
        a = (17, 40)
        l = (1, 2)
        d1 = (80, 100)
        d2 = (120, 150)
        t1 = (70, 100)
        t2 = (70, 90)
        t3 = (80, 110)

        # set modeling string
        sa = f"&(rand({a[0]}, {a[1]}))"
        sf = f"F(rand({l[0]}, {l[1]}))"
        sd1 = f"/(rand({d1[0]}, {d1[1]}))"
        sd2 = f"/(rand({d2[0]}, {d2[1]}))"
        st1 = f"/(rand({t1[0]}, {t1[1]}))"
        st2 = f"/(rand({t2[0]}, {t2[1]}))"
        st3 = f"/(rand({t3[0]}, {t3[1]}))"
        # 互生
        b_m = f"{sf}[:^(65){L};]{sf}[:&(65){L};]{sf}"
        # 対生
        b_o = f"{sf}[:^(65){L};][:&(65){L};]{sf}"
        # 十字対生
        b_co = f"{sf}[:^(65){L};][:&(65){L};]{sf}[:/(90)^(65){L};][:/(90)&(65){L};]{sf}"
        # 輪生
        b_r = f"{sf}[:^(65){L};][:&(65){L};][:/(90)^(65){L};][:/(90)&(65){L};]{sf}"

        lsys.delete()

        # define lsystem
        lsys.define("lr", "1.109")
        lsys.define("vr", "1.732")

        lsys.set_axiom(f"p(skin)¤(0.1)F(0.0001)F(1)/(45)X")
        lsys.add_rule("X", f"¤(mul(0.1,vr))F(4)[{sa}{b_o}Y]{sd1}[{sa}{b_o}Y]{sd2}[{sa}{b_o}Y]")
        lsys.add_rule("X", f"¤(mul(0.1,vr))F(4)[{sa}{b_o}Y]{st1}[{sa}{b_o}Y]{st2}[{sa}{b_o}Y]{st3}[{sa}{b_o}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4)[{sa}{b_co}Y]{sd1}{sd2}[{sa}{b_co}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4){sd1}[{sa}{b_co}Y]{sd2}[{sa}{b_co}Y]")
        lsys.add_rule("Y", f"¤(mul(0.1,vr))F(4)[{sa}{b_co}Y]{sd1}[{sa}{b_co}Y]{sd2}")
        lsys.add_rule("F(l)", "F(mul(l,lr))")
        lsys.add_rule("¤(w)", "¤(mul(w,vr))")

    return lsys


def main(lsys, seed, dtype, data_idx, camera_name, data_dir, img_res, render_num, r):
    if dtype == 'ara':
        lsys.exec(min_iterations=1, angle=60, seed=seed)
    elif dtype == 'komatsuna':
        lsys.exec(min_iterations=1, angle=75, seed=seed)
    else:
        lsys.exec(min_iterations=2, seed=seed)

    obj_name = []

    if dtype == 'plant':
        obj = lsys.objects[0]
        leaves = lsys.objects[1:]

        # change color
        obj_name.append(obj.name)
        tree = bpy.data.objects.get(obj.name)
        bpy.context.view_layer.objects.active = tree
        tree_height = get_height_branch(tree) - 5
        if not tree.data.vertex_colors:
            tree.select_set(True)
            bpy.ops.mesh.vertex_color_add()
        vc = tree.data.vertex_colors[0].data
    elif dtype == 'amodal':
        obj = lsys.objects[0]
        leaves = lsys.objects[1:]

        # change color
        obj_name.append(obj.name)
        tree = bpy.data.objects.get(obj.name)
        bpy.context.view_layer.objects.active = tree
        tree_height = get_height_branch(tree) + 15
        if not tree.data.vertex_colors:
            tree.select_set(True)
            bpy.ops.mesh.vertex_color_add()
        vc = tree.data.vertex_colors[0].data
    elif dtype == 'komatsuna':
        leaves = lsys.objects
        tree_height = 8
    else:
        leaves = lsys.objects
        tree_height = 5

    for i in range(0, len(leaves)):
        obj_name.append(leaves[i].name)

    bpy.ops.export_mesh.ply(filepath=f'{data_dir}/model/plant_{str(seed).zfill(7)}.ply')

    json_data = {'camera_position': [],
                 'num_leaves': len(leaves),
                 'seed_value': seed,
                 'turtle_string': lsys.turtle_string}
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.name = camera_name
    render_plant(plant_idx=seed, cam_name=camera_name, res=img_res, photo_num=render_num, radius=r, img_path=data_dir,
                 dic=json_data, parent_pos=tree_height, obj_name=obj_name, dtype=dtype)
    active_object = bpy.context.active_object
    if active_object.name == camera_name:
        bpy.ops.object.delete()
    with open(f'{data_dir}/json/plant_{str(seed).zfill(7)}.json', 'w') as out_file:
        json.dump(json_data, out_file, indent=4)

    objects = bpy.data.objects

    # すべてのオブジェクトを削除
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == '__main__':
    # environmet value
    # DATA_TYPE : 'amodal' (including amodal data), 'leaf' (for supervised data), 'plant' (for self-supervised data)
    dtype_list = ['amodal', 'plant', 'komatsuna', 'ara']
    DATA_TYPE = 'ara'
    assert DATA_TYPE in dtype_list, f'{DATA_TYPE} is not in {dtype_list}'
    gen_time = len(os.listdir('plant_data')) + 1
    os_dir = os.path.join('plant_data', f'data_{str(gen_time).zfill(5)}')
    root_dir = f''
    os.mkdir(os_dir)
    os.mkdir(os.path.join(os_dir, 'img'))
    os.mkdir(os.path.join(os_dir, 'binary'))
    os.mkdir(os.path.join(os_dir, 'model'))
    os.mkdir(os.path.join(os_dir, 'json'))
    CAM_NAME = "Camera"
    # number of rendering per object
    RENDER_NUM = 1
    RESOLUTION = 450
    if DATA_TYPE == 'amodal':
        os.mkdir(os.path.join(os_dir, 'amodal_seg'))
        RADIUS = 0
    elif DATA_TYPE == 'ara' or 'komatsuna':
        RADIUS = 0.1
    else:
        RADIUS = 25
    DATA_NUM = 1

    int_list = [i for i in range(1, 1000001)]
    seed_list = random.sample(int_list, DATA_NUM)
    if DATA_TYPE == 'plant' or 'amodal':
        leaf_str = create_leaf_string(DATA_TYPE)
        lsys = create_lsystem(DATA_TYPE, leaf_str)
    for idx, seed in enumerate(seed_list):
        if DATA_TYPE == 'komatsuna':
            matured = random.randint(0, 1)
            sangle_list, bangle_list = [], []
            if matured:
                sleaf_num = random.randint(0, 1)
                bleaf_num = random.randint(3, 4)
                if sleaf_num:
                    angle_list = [i for i in range(85, 95)]
                    sangle_list.append(random.sample(angle_list, 1))
                bangle_intval = 340 / bleaf_num
                for i in range(bleaf_num):
                    bangle_list.append(
                        random.randint(int(bangle_intval * (i + 1)) - 20, int(bangle_intval * (i + 1)) + 20))
            else:
                sleaf_num = random.randint(1, 4)
                bleaf_num = 0
                sangle_intval = 340 / sleaf_num
                for i in range(sleaf_num):
                    sangle_list.append(
                        random.randint(int(sangle_intval * (i + 1)) - 20, int(sangle_intval * (i + 1)) + 20))

            leaf_str_list = create_leaf_string(DATA_TYPE)
            leaf_dict = {
                "string": leaf_str_list,
                "matured": matured,
                "sleaf_num": sleaf_num,
                "bleaf_num": bleaf_num,
                "sangle_list": sangle_list,
                "bangle_list": bangle_list
            }
            lsys = create_lsystem(DATA_TYPE, leaf_dict)
        elif DATA_TYPE == 'ara':
            leaf1_num = random.randint(7, 10)
            leaf2_num = random.randint(8, 10)
            angle1_list, angle2_list = [], []
            angle1_intval = 340 / leaf1_num
            angle2_intval = 340 / leaf2_num
            for i in range(leaf1_num + 1):
                if i == 0:
                    angle1_list.append(random.randint(0, 20))
                else:
                    angle1_list.append(random.randint(int(angle1_intval*i)-20, int(angle1_intval*i)+20))
            for j in range(leaf2_num + 1):
                if j == 0:
                    angle2_list.append(random.randint(0, 20))
                else:
                    angle2_list.append(random.randint(int(angle2_intval*j)-20, int(angle2_intval*j)+20))
            leaf_str_list = create_leaf_string(DATA_TYPE)
            leaf_dict = {
                "string": leaf_str_list,
                "leaf1_num": leaf1_num,
                "leaf2_num": leaf2_num,
                "angle1_list": angle1_list,
                "angle2_list": angle2_list
            }
            lsys = create_lsystem(DATA_TYPE, leaf_dict)
        main(lsys, seed, DATA_TYPE, idx, CAM_NAME, root_dir, RESOLUTION, RENDER_NUM, RADIUS)
