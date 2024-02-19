# for num in {100..300}
for num in {100..101}
do 
    docker run --runtime=nvidia --rm --interactive \
                --user $(id -u):$(id -g)        \
                --gpus all                    \
                --env KUBRIC_USE_GPU=1          \
                --volume "$(pwd):/kubric"       \
                kubricdockerhub/kubruntu        \
                python3 sim_render.py            \
                --data_dir=output/convert       \
                --output_dir=/mnt/sdb/data/cs/physics_super_clevr/       \
                --camera=fixed \
                --height=realistic \
                --iteration=$num \
                --scene_size 5 
done

for num in {102..103}
do 
    docker run --runtime=nvidia --rm --interactive \
                --user $(id -u):$(id -g)        \
                --gpus all                    \
                --env KUBRIC_USE_GPU=1          \
                --volume "$(pwd):/kubric"       \
                kubricdockerhub/kubruntu        \
                python3 sim_render.py            \
                --data_dir=output/convert       \
                --output_dir=/mnt/sdb/data/cs/physics_super_clevr/       \
                --camera=fixed \
                --height=random \
                --iteration=$num \
                --scene_size 5 
done