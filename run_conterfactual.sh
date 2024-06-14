

time="$(date +%Y-%m-%d_%H-%M-%S)"

# for num in {0..200}
for num in {1001..1100}
do 
    CUDA_VISIBLE_DEVICES=0 python sim_render_conterfacual.py \
        --data_dir=assets \
        --job-dir=output/output_v3_counterfacual \
        --scratch_dir=output/tmp/tmp-$time \
        --camera=fixed \
        --height=realistic \
        --iteration=$num \
        --scene_size 5 \
        --load_scene=output/output_v3_1k
done

