time="$(date +%Y-%m-%d_%H-%M-%S)"
for num in {0..100}
do 
    CUDA_VISIBLE_DEVICES=xx python sim_render_color_defined_load_scene.py \
        --data_dir=assets \
        --job-dir=output/superclevr-physics \
        --scratch_dir=output/tmp/tmp-$time \
        --camera=fixed \
        --height=realistic \
        --iteration=$num \
        --scene_size 5 
done