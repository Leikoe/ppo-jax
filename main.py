import gymnasium as gym
import jax, distrax, optax
from flax import nnx
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

orthogonal = nnx.nn.initializers.orthogonal
constant = nnx.nn.initializers.constant

BATCH_SIZE = 64
MAX_STEPS = 256
ITERATIONS = 50
K = 5
EPS = 0.1
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95


class Model(nnx.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, dropout=0.1, rngs: nnx.Rngs=nnx.Rngs(0)):
        self.policy = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, num_actions, rngs=rngs, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        )
        self.value = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nnx.Dropout(dropout, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, 1, rngs=rngs, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        )

    def __call__(self, x: jax.Array) -> tuple[distrax.Categorical, jax.Array]:
        logits = self.policy(x)
        return distrax.Categorical(logits), self.value(x)

@nnx.jit
def get_action_logprobs_value(model: Model, obs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    pi, value = model(obs)
    action = pi.sample(seed=key)
    return action, pi.log_prob(action), value

@nnx.jit
def get_action(model: Model, obs: jax.Array, key: jax.Array) -> jax.Array: return model(obs)[0].sample(seed=key)

# @nnx.jit
def loss_fn(model, observations, actions, values, actions_log_probs, advantages, returns):

    observations = jax.lax.stop_gradient(observations)
    actions = jax.lax.stop_gradient(actions)
    values = jax.lax.stop_gradient(values)
    actions_log_probs = jax.lax.stop_gradient(actions_log_probs)
    advantages = jax.lax.stop_gradient(advantages)
    returns = jax.lax.stop_gradient(returns)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pi, value = model(observations)
    value = value.flatten()

    prob_ratio = jnp.exp(pi.log_prob(actions) - actions_log_probs) # = p_new(a_t) / p_old(a_t) because of log rules log(a/b) = log(a) - log(b)

    policy_loss = -jnp.mean(jnp.minimum( # loss is -L_CLIP because we want to maximize L_CLIP but using gradient descent
        prob_ratio * advantages, jnp.clip(prob_ratio, min=1.0-EPS, max=1.0+EPS) * advantages
    ))
    value_loss = jnp.mean(jnp.square(value - returns))
    entropy_loss = -jnp.mean(pi.entropy())
    return policy_loss + 0.5 * value_loss + 0.0 * entropy_loss

def calculate_gae_returns(rewards, values, dones, last_value):
    """ adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py#L142 """
    def _get_advantages(gae_and_next_value, reward_value_done):
        gae, next_value = gae_and_next_value
        reward, value, done = reward_value_done
        delta = reward + DISCOUNT_FACTOR * next_value * (1 - done) - value
        gae = delta + DISCOUNT_FACTOR * GAE_LAMBDA * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True, # scanning the trajectory batch in reverse order
    )
    return advantages, advantages + values # advantages + values = returns

SEED = 7
key = jax.random.key(SEED)
env = gym.make("LunarLander-v3")
model = Model(8, 128, 4) # lunar lander has obs (8,) and action(1,) in [0,3] range
optimizer = nnx.Optimizer(model,optax.chain(
  optax.clip(0.5),
  optax.adamw(1e-3),
))  # reference sharing

value_and_grad_loss_fn = nnx.value_and_grad(loss_fn)

for iteration in range(ITERATIONS):
    print(f"{iteration=}")
    observations = []
    value_estimates = []
    actions = []
    actions_log_probs = []
    rewards = []
    dones = []

    done_rewards = []

    # single actor for now so no loop
    # Run policy πθold in environment for T timesteps
    observation, _ = env.reset()
    for step in tqdm(range(MAX_STEPS)):
        key, subkey = jax.random.split(key)
        action, log_probs, value = get_action_logprobs_value(model, observation, subkey)
        action, value = action.item(), value.item()

        observations.append(observation)
        value_estimates.append(value)
        actions.append(action)
        actions_log_probs.append(log_probs)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        dones.append(terminated or truncated)
        if terminated or truncated:
            done_rewards.append(reward)
            observation, _ = env.reset()
    key, subkey = jax.random.split(key)
    last_value = get_action_logprobs_value(model, observation, subkey)[2].item() # one more required for gae's deltas

    # Compute advantage estimates Â_1,..., Â_T
    # deltas: list[float] = [r + DISCOUNT_FACTOR * (1 - d) * nv - v for r, v, nv, d in zip(rewards, value_estimates[:-1], value_estimates[1:], dones)]
    # gaes: list[float] = [deltas[-1]]
    # for delta, done in reversed(list(zip(deltas[:-1], dones))):
    #     discounted_last = 0. if done else DISCOUNT_FACTOR * GAE_LAMBDA * gaes[0]
    #     gaes.insert(0, delta + discounted_last)
    # advantages = gaes

    observations = jnp.array(observations)
    value_estimates = jnp.array(value_estimates)
    actions = jnp.array(actions)
    actions_log_probs = jnp.array(actions_log_probs)
    rewards = jnp.array(rewards)
    dones = jnp.array(dones)
    # print(observations.shape, value_estimates.shape, actions.shape, rewards.shape, dones.shape)

    # advantages = jnp.array(advantages)
    # returns = advantages + value_estimates

    advantages, returns = calculate_gae_returns(rewards, value_estimates, dones, last_value)

    print("iter mean reward", jnp.mean(rewards))
    print(f"done rewards {done_rewards}")

    # Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤NT
    for epoch in range(K):
        # print(f"{epoch=}")
        for batch in range(len(observations) // BATCH_SIZE):
            key, _key = jax.random.split(key)
            batch_idxs = jax.random.randint(_key, shape=(BATCH_SIZE,), minval=0, maxval=len(observations))
            loss, grads = value_and_grad_loss_fn(model, observations[batch_idxs], actions[batch_idxs], value_estimates[batch_idxs], actions_log_probs[batch_idxs], advantages[batch_idxs], returns[batch_idxs])
            optimizer.update(grads)  # inplace updates
env.close()

# # EVAL
# env = gym.make("LunarLander-v3", render_mode="human")

# observation, _ = env.reset(seed=42)
# for _ in range(1000):
#     key, subkey = jax.random.split(key)
#     action = get_action(model, observation, subkey).item()
#     observation, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated: observation, _ = env.reset()

# env.close()
