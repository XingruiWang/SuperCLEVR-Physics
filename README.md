# SuperClEVR-physics

# Setup

1. python version 3.10

2. Install `kubric`

    See link https://github.com/google-research/kubric/issues/100

    Or

    ```
    git clone https://github.com/google-research/kubric.git
    cd kubric
    pip install -r requirements.txt
    pip install .
    ```

3. `pip install bpy==3.5` (or 3.4)

4. `pip install pybullet`

5. Maybe need someother packages

    ```sh
    pip install singledispatchmethod
    pip install OpenEXR
    ```

# Generate the code

1. Generate the first camera view with annotation, id from 0 to 100.

    The output is `output/physics_super_clevr_c0`.
    
    ```
    for num in {0..100}
    do 
        python sim_render_color_defined_load_scene.py \
            --data_dir=assets \
            --job-dir=output/physics_super_clevr_c0 \
            --camera=random \
            --height=realistic \
            --iteration=$num \
            --scene_size 5
    done
    ```

    The `camera` will be ramdonly sample in sphere. It can also be fixed by setting `camera=fixed`, and change the position in line 80 in `sim_render_color_defined_load_scene.py`

2. Regenerate 

    By setting `load_scene=output/physics_super_clevr_c0`, it will load the generated scene from camera 0, and set a new camera c1. 
    
    ```
    for num in {0..1}
    do 
        python sim_render_color_defined_load_scene.py \
            --data_dir=assets \
            --job-dir=output/physics_super_clevr_c1 \
            --camera=random \
            --height=realistic \
            --iteration=$num \
            --scene_size 5 \
            --load_scene output/physics_super_clevr_c0
    done
    ```

Weird motion:

/home/angtian/xingrui/superclevr2kubric/output_1k/super_clever_700


---


1. Convert obj

- construct folder 

E.g. bicycle / objs / bicycle.obj

- run convert.sh

generate both 
Requirement:
Node >= 14.0


/home/shuo/blender-2.79b-linux-glibc219-x86_64/blender --background \
    --python super_clevr_renderer.py -- \
    --convert_dir /home/angtian/xingrui/superclevr2kubric/assets/bicycles \
    --make_kubric 1 \
    --make_glb 1