for num in {62..91}
do 
    sudo cp -r /home/shuo/kubric_exp/superclevr2kubric/output/super_clever_$num /mnt/sdb/data/cs/output_new
done

# for num in {84..91}
# do 
#     sudo rmdir /mnt/sdb/data/cs/output_new/super_clever_$num 
# done