# for num in {25..49}
# for num in {25..49}
# for num in {50..74}
# for num in {26..50}
# for num in {101..200}
# for num in {200..250}
# for num in {251..300}
# for num in {421..500}
# for num in {195..199}
# for num in {0..10}
# do 
#     docker run --runtime=nvidia --rm --interactive \
#                 --user $(id -u):$(id -g)        \
#                 --gpus all                    \
#                 --env KUBRIC_USE_GPU=1          \
#                 --volume "$(pwd):/kubric"       \
#                 kubricdockerhub/kubruntu        \
#                 python3 sim_render_color_defined_load_scene.py            \
#                 --data_dir=output/convert       \
#                 --camera=fixed \
#                 --height=realistic \
#                 --iteration=$num \
#                 --scene_size 5 \
#                 --skip_segmentation
# done

# time="$(date +%Y-%m-%d_%H-%M-%S)"
# for num in {700..800}
# do 
#     CUDA_VISIBLE_DEVICES=4 python sim_render_color_defined_load_scene.py \
#         --data_dir=assets \
#         --job-dir=output/output_v3_1k \
# 	    --scratch_dir=output/tmp/tmp-$time \
#         --camera=fixed \
#         --height=realistic \
#         --iteration=$num \
#         --scene_size 5
# done
time="$(date +%Y-%m-%d_%H-%M-%S)"
for num in {0..100}
do 
    CUDA_VISIBLE_DEVICES=0 python sim_render_color_defined_load_scene.py \
        --data_dir=assets \
        --job-dir=output/output_test \
	    --scratch_dir=output/tmp/tmp-$time \
        --camera=fixed \
        --height=realistic \
        --iteration=$num \
        --scene_size 5
done