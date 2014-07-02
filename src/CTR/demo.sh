#! /bin/bash

echo "First, convert raw data format to satisfy the requriements of CTR!"
python convertDataFormat.py -d 0
#python convertDataFormat.py -d 1

echo "Second, running lda to get topic distribution and word distribution results!"
lda est 0.01 50 settings.txt ./doubanBeijingEventInfo.dat random ./lda_results
#lda est 0.01 50 settings.txt ./doubanShanghaiEventInfo.dat random ./lda_results

echo "Third, running CTR model to obtain model parameters!"
./ctr --directory ./ --user ./doubanBeijingUserRecords.dat --item ./doubanBeijingEventRecords.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 --mult ./doubanBeijingEventInfo.dat --num_factors=50 --save_lag 20 --max_iter 200 --theta_init ./lda_results/final.gamma --beta_init ./lda_results/final.beta
#./ctr --directory ./ --user ./doubanShanghaiUserRecords.dat --item ./doubanShanghaiEventRecords.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 --mult ./doubanShanghaiEventInfo.dat --num_factors=50 --save_lag 20 --max_iter 200 --theta_init ./lda_results/final.gamma --beta_init ./lda_results/final.beta

echo "Finally, do personalized event recommendation based on previous results!"
python predict.py -d 0
#python predict.py -d 1

