convert -delay 12 -loop 0 /home/angtian/xingrui/superclevr2kubric/output/super_clever_100/rgba_*.png output1.gif 
# ffmpeg -framerate 30 -i /home/angtian/xingrui/superclevr2kubric/output/super_clever_100/rgba_%05d.png -c:v libx264 -pix_fmt yuv420p /home/angtian/xingrui/superclevr2kubric/output/super_clever_100/output.mp4
