# Q1

The law of iterated expectations is: the expectation of a conditional expectation, is the same as the unconditional expectation.

So you can break apart the expectation of the policy trajectory, to the expectation of the expectation of the state-action marginal times the trajectory conditioned on (s, a).

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q1_unconditioned.jpg)

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q1.jpg)

b) It is the same because in the law of iterated expectations --- the expectation of a conditional expectation, is the same as the unconditional expectation. Which will be the p theta expectation which will equal 1 since it is a distribution.

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

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q5.png)

# Q7

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q7.png)

# Q8

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q8_bs_lr.png)

1) How did the batch size and learning rate affect the performance?

The learning rate had the largest affect on performance because all batch sizes of lr 0.02 outperformed all other lr / batch sizes.

Then within lr 0.02 the performance increased with batch size.

2) Provide a single plot plotting the learning curves for all four runs.

![](https://raw.githubusercontent.com/jperl/rl-homework/master/berkeley/hw2/answers/q8.png)
