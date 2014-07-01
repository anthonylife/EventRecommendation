#!/bin/sh

temp='temp'
if [ $temp$2 = $temp ];
then
    echo 'Commander: ./demo.sh -m <int> (Choose which data to run.)'
    exit 1
fi


if [ $2 = 0 ];
then
    echo 'Running on douban Beijing dataset'
    python createTrainingInstance.py -d 0
    python train.py -d 0 -m 0
    python predict.py -d 0
elif [ $2 = 1 ];
then
    echo 'Running on douban Shanghai dataset'
    python createTrainingInstance.py -d 1
    python train.py -d 1 -m 0
    python predict.py -d 1
else
    echo 'Invalid choice of dataset'
    exit 1
fi

