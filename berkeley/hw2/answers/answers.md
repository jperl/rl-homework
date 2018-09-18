# Q4

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q4_sb_cartpole.png)

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q4_lb_cartpole.png)

1) Which gradient estimator has better performance without advantage-centering— the trajectory-centric one, or the one using reward-to-go?

Reward-to-go. It has the higher performance both with and without advantage normalization.

2) Did advantage centering help?

Yes. It is extremely helpful in the small batch case. Note, you cannot use it on the non-reward-to-go, since it's std is 0.

3) Did the batch size make an impact?

Yes. The reward is more stable, especially when there is not advantage normalization.

# Q5

Params in [q5.sh](https://github.com/jperl/rl-homework/blob/master/berkeley/hw2/q5.sh)

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q5.png)
