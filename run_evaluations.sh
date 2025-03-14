#! /bin/bash

metrics=("mae" "mse")
target_properties=("mass" "thickness" "e1")
regressors=("linear" "ridge" "sgd" "svr" "tree" "forest" "xgb")

for regressor in "${regressors[@]}"; do

    for metric in "${metrics[@]}"; do

        for target_property in "${target_properties[@]}"; do

            python ./estimator.py -m $regressor -t $target_property -e $metric

        done

    done

done