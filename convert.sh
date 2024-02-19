CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/shuo/blender-2.79b-linux-glibc219-x86_64/blender --background \
    --python super_clevr_renderer.py -- \
    --shape_dir /mnt/sdb/data/cs/CGPart \
    --model_dir data/save_models_1/ \
    --properties_json data/properties_cgpart.json \
    --convert_dir ../output/convert \
    --convert 0 \
    --fix_watertight 0 \
    --make_obj 0 \
    --make_glb 0 \
    --make_kubric 1 \
    --iteration 20

