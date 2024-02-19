convert -delay 12 -loop 0 /home/shuo/kubric_exp/superclevr2kubric/output/rgba_*.png output1.gif 
ffmpeg -framerate 30 -i /home/shuo/kubric_exp/superclevr2kubric/output/super_clever_60/rgba_%05d.png -c:v libx264 -pix_fmt yuv420p /home/shuo/kubric_exp/superclevr2kubric/output/super_clever_60/output.mp4
