#!/bin/bash
set -e

cd detector_py3
maxeps=999
f=3
for (( i=1; i<=$maxeps; i+=1)) 
do
    echo "process $i epoch"
	
	if [ $i -lt 10 ]; then
	    python main.py --model dpn3d26 -b 1 --resume results/dpn3d26/retrft96$f/CkptFile/00$i.ckpt --test 1 --save-dir dpn3d26/retrft96$f/ --config config_trainingpy3$f --gpu=2
	elif [ $i -lt 100 ]; then 
	    python main.py --model dpn3d26 -b 1 --resume results/dpn3d26/retrft96$f/CkptFile/0$i.ckpt --test 1 --save-dir dpn3d26/retrft96$f/ --config config_trainingpy3$f --gpu=2
	elif [ $i -lt 1000 ]; then
	    python main.py --model dpn3d26 -b 1 --resume results/dpn3d26/retrft96$f/CkptFile/$i.ckpt --test 1 --save-dir dpn3d26/retrft96$f/ --config config_trainingpy3$f --gpu=2
	else
	    echo "Unhandled case"
    fi

    if [ ! -d "results/dpn3d26/retrft96$f/val$i/" ]; then
        mkdir results/dpn3d26/retrft96$f/val$i/
    fi
    mv results/dpn3d26/retrft96$f/bbox/*.npy results/dpn3d26/retrft96$f/val$i/
done 
