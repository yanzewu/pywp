date
cd ..
python -m pywp --potential test.Tully1 -L 16 --Ly 8 -M 256 --My 128 --init_x -5 --init_y 0 --sigma_x 1 --sigma_y 1 --init_px 20 --init_py 0 --init_s 0 --Nstep 30000 --mass 2000 --dt 0.05 --output_step 1000 --gpu=false --traj test/tully1.trj > test/out-tully1.dat
date
