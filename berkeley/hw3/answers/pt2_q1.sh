# 1 target update, 1 gradient step
# ----------------------------------------
# |               Time |            38.2 |
# |          Iteration |              99 |
# |      AverageReturn |             9.3 |
# |          StdReturn |           0.797 |
# |          MaxReturn |              11 |
# |          MinReturn |               8 |
# |          EpLenMean |             9.3 |
# |           EpLenStd |           0.797 |
# | TimestepsThisBatch |           1e+03 |
# |     TimestepsSoFar |        1.01e+05 |
# ----------------------------------------
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1

# 100 target updates, 1 gradient step
# ----------------------------------------
# |               Time |            66.3 |
# |          Iteration |              99 |
# |      AverageReturn |            9.43 |
# |          StdReturn |           0.699 |
# |          MaxReturn |              11 |
# |          MinReturn |               8 |
# |          EpLenMean |            9.43 |
# |           EpLenStd |           0.699 |
# | TimestepsThisBatch |        1.01e+03 |
# |     TimestepsSoFar |        1.01e+05 |
# ----------------------------------------
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1

# 1 target updates, 100 gradient steps
# ----------------------------------------
# |               Time |            53.6 |
# |          Iteration |              99 |
# |      AverageReturn |            9.26 |
# |          StdReturn |           0.722 |
# |          MaxReturn |              11 |
# |          MinReturn |               8 |
# |          EpLenMean |            9.26 |
# |           EpLenStd |           0.722 |
# | TimestepsThisBatch |        1.01e+03 |
# |     TimestepsSoFar |        1.01e+05 |
# ----------------------------------------
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100

# 10 target updates, 10 gradient steps
# ----------------------------------------
# |               Time |              54 |
# |          Iteration |              99 |
# |      AverageReturn |            8.81 |
# |          StdReturn |           0.511 |
# |          MaxReturn |              10 |
# |          MinReturn |               8 |
# |          EpLenMean |            8.81 |
# |           EpLenStd |           0.511 |
# | TimestepsThisBatch |           1e+03 |
# |     TimestepsSoFar |        1.01e+05 |
# ----------------------------------------
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10
