
p=12
cd ..
python -m pywp --potential test.Flat --potential_params 0.05,3,5 -L 24 -M 256 --init_x -3 --init_y -3 --sigma_x 1 --sigma_y 1 --init_px $p --init_py $p --init_s 1 --Nstep 30000 --mass 1000 --dt 0.05 --output_step 1000 --gpu=true --traj test/traj-flat-p$p.trj --output=2 > test/out-flat-p$p.dat

