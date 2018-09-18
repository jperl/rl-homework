batch_sizes=( 10000 30000 50000 )
learning_rates=( 0.005 0.01 0.02 )

for bs in "${batch_sizes[@]}"
do
  for lr in "${learning_rates[@]}"
  do
  	experiment="python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $bs -lr $lr -rtg --nn_baseline --exp_name hc_b${bs}_r${lr}"
    echo "Running $experiment"
    eval $experiment
  done
done
