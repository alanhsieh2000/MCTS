# Monte Carlo Tree Search
Monte Carlo Tree Search (MCTS) is a rollout algorithm, the goal of which is to improve its policy. The root node of the tree is the current 
state, and child nodes of a state node are all possible actions. Alternatingly, the child node of an action node is a state node, which is the 
next state after taking the action represented by its parent action node. Every time when we start a new episode, the evironment may change 
to a new initial state. The root node is then changed, and a new iteration starts. Every iteration consists of 4 sequential operations: selection, 
expansion, simulation, and backup. MCTS has 2 policies, tree policy and rollout policy.

1. Selection
    - If there is no unseen child action node, all actions ever taken in the state, MCTS uses the tree policy to select an action to take. 
    The simplest tree policy is random. The value of each action node is the estimated action-value, $Q(s, a)$ for the action node and its parent 
    state node pair, so the greedy tree policy would select the action by $\arg \max_a Q(s, a)$.
2. Expansion
    - If there is any unseen child action node, the tree is expanded by taking one of them randomly and adding the taken action as a child action 
    node.
3. Simulation
    - Starting from the expanded node, MCTS becomes the Monte Carlo learning. It uses the rollout policy, random, to sample $s_{t+1}$ 
    and $r_{t+1}$ until a terminal state, or truncated.
4. Backup
    - Once a terminal state is reached, the value of the expanded action node, $Q(s, a)$, is updated as the discounted accumulated rewards. 
    And then, back up to the action node just one level under the root node, all values of action node on the path are updated.

MCTS does not need to evaluate values of all states or action-state pairs. When the time for learning is up, the agent stops MCTS iterations and 
takes the action by $\arg \max_a Q(s, a)$ until the bottom of the tree, a leaf action node. At this time, the agent switches back to the MCTS 
mode, and the current state becomes the root node.

# Upper Confidence Tree
Upper Confidence bounds applied to Trees (UCT) treats its selection operation as a multi-armed bandit problem and uses UCB algorithm. 

$${a_t=\arg\max_{a} Q_t(a)+U_t(a)}$$
$${N_t(a)=\begin{cases}
  N_{t-1}(a)+1, & \text{if $A_t=a$}\\
  N_{t-1}(a), & \text{otherwise}
\end{cases}}$$
$${U_{t+1}(a)=c \cdot \sqrt{\frac{\log t}{N_t(a)}}}$$

    function UCT_sample(state_node, depth)
      if isTerminal(state_node):
        return score(state_node)
      if depth + 1 == MAX_DEPTH:
        return evaluate(state_node)
      if all children expanded:
        action_node = UCB_sample(state_node)
        next_state_node, next_reward = take_action(state_node, action_node)
        outcome = UCT_sample(next_state_node, depth+1)
      else: # expansion
        action = random sampling from unexpanded children actions
        action_node = create(action)
        add action_node to the tree
        next_state_node, next_reward = take_action(state_node, action_node)
        outcome = random_playout(next_state_node)
      q = reward + gamma * outcome
      update_value(state_node, action_node, q)
      return q

    # selection
    function UCB_sample(state_node)
      weights = array(size = state_node.children_count)
      for i in range(state_node.children_count):
        action_node = state_node.children[i]
        weights[i] = action_node.value + C * sqrt(ln(state_node.visits) / action_node.visits)
      distribution = normalize weights
      return state_node.children[distribution.sample()]

    # simulation
    function random_playout(state_node)
      s = state_node
      outcome = 0.0
      rewards = []
      while not isTerminal(s):
        a = random sampling from s.children
        s, r = take_action(s, a)
        rewards.append(r)
      rewards.append(score(s))
      i = 0
      for i in range(len(rewards)):
        outcome += power(gamma, i) * rewards[i]
      return outcome

    # backup
    function update_value(state_node, action_node, value)
      state_node.visits ++
      action_node.visits ++
      action_node.value += (value - action_node.value) / action_node.visits

# Reference
- Carnegie Mellon University, Fragkiadaki, Katerina, et al. 2024. "10-403 Deep Reinforcement Learning" As of 8 November, 2024. 
https://cmudeeprl.github.io/403website_s24/.
- Sutton, Richard S. and Barto, Andrew G. 2018. Reinforcement Learning - An indroduction, second edition. The MIT Press.
- Kocsis, L. and Szepesvári, C. 2006. Bandit based Monte-Carlo planning. In Machine Learning: European Conference on Machine Learning 2006, Springer.
