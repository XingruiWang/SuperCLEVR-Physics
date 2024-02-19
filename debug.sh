docker run --runtime=nvidia --rm --interactive \
            --user $(id -u):$(id -g)        \
            --gpus all                    \
            --env KUBRIC_USE_GPU=1          \
            --volume "$(pwd):/kubric"       \
            kubricdockerhub/kubruntu        \
            python3 sim_render.py            \
            --data_dir=output/convert       \
            --output_dir=/mnt/sdb/data/cs/physics_super_clevr/       \
            --camera=fixed              \
            --iteration=102