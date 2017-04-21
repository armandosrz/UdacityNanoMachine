# Capstone Project

## Reinforcement learning in Conniption

Conniption is a modified connect-four game, in which the player has the possibility of flipping the board upside down. When a flip occurs, the chips falls down creating new combinations, and in some cases a winning solution. In each player's turn the player decides to flip or not, place and again to flip or not. Each player is allowed to flip four times during a game and no consecutive flips between players are allowed. The rest of the game remains the same, a user place a chip from above in a 7x6 grid and finished his turn. A second player repeats the process in his turn. The winner has to link 4 consecutive chips in any possible direction.

In this project we will use reinforcement learning to make and agent play conniption, with the final goal to defeat the agents build upon mini-max with alpha-beta pruning.
a. Task:  Playing Conniption
b. Performance:  Percent of games won against other players. Due to the characteristics of the game it matters whether you start as the player one or player Two.
The main goal will be to build a reinforcement learning agent that will always maximize the next movement in order to obtain the best possible move. We will analyze the results from using a predictive model (minimax algorithm) against learning through repetition (reinforcement learning). Connect 4 is a deterministic solved game in which under optimum play from both players, player one will always win. We will explore if the same behavior is translated into Conniption and the influence it has on the agents.

For each specific part of the project please refer to:

 - [Final Report](CapstoneProject.pdf)
 - [Repository](/Conniption)
 - [Proposal](CapstoneProposal.pdf)
 - [Proposal Submission](https://review.udacity.com/#!/reviews/396992)

 For the necessary project libraries, read the Conniption [README](/Conniption/README.md).
