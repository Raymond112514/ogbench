"""Online AWR on OGBench: warm-start on the play dataset, then collect / relabel / retrain.

Same play dataset + joint training recipe as awr/offline.py (--advantage {iql, classifier}),
made online by repeating, for `--max_rounds` rounds:

  1. Collect --episodes_per_round episodes in the real env with the current policy.
  2. If --advantage classifier, label the new transitions with the oracle.
  3. Add the new data into the same growing buffer (original play data + all rounds so far).
  4. Continue training the *same* agent (not reinitialized) for --iql_steps_per_round joint
     gradient steps ("train IQL"/"train classifier").
  5. Continue training the *same* agent for --policy_steps_per_round more joint gradient
     steps ("train policy").

Steps 4-5 both call the identical joint update used by awr/offline.py (advantage + actor
are always updated together, per the fixed AWR recipe) — they are two sequential calls to
the same trainer so the round structure matches the spec; together they add
`iql_steps_per_round + policy_steps_per_round` joint gradient steps per round.

Examples:
  python awr/online_ogbench.py --env_name=cube-single-play-singletask-task1-v0 --advantage iql
  python awr/online_ogbench.py --env_name=cube-single-play-singletask-task1-v0 \\
      --advantage classifier --chunk_size 4 --annotated_path awr/data/task1_oracle.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')


def collect_episodes(
    agent, env, num_episodes: int, chunk_size: int, seed: int, temperature: float = 1.0, record_sim_state: bool = False,
) -> tuple[dict, float]:
    """Roll out `num_episodes` with the current agent; return a rollout dict + success rate.

    The rollout dict matches awr.iql.dataset.transitions_from_data's expected input
    (per-step observations/actions/next_observations/successes/episode_ends/chunk_size),
    optionally augmented with per-step qpos/qvel for oracle relabeling.
    """
    import jax

    act_dim = int(np.prod(env.action_space.shape))
    key = jax.random.PRNGKey(seed)
    observations, actions, next_observations, successes = [], [], [], []
    qpos_list, qvel_list = [], []
    episode_ends = []

    for _ in range(num_episodes):
        ob, info = env.reset()
        done = False
        while not done:
            key, sample_key = jax.random.split(key)
            flat = np.array(agent.sample_actions(observations=ob, seed=sample_key, temperature=temperature))
            chunk = flat.reshape(chunk_size, act_dim) if chunk_size > 1 else flat.reshape(1, -1)
            for a in chunk:
                if done:
                    break
                if record_sim_state:
                    u = env.unwrapped
                    qpos_list.append(np.array(u.data.qpos, dtype=np.float64))
                    qvel_list.append(np.array(u.data.qvel, dtype=np.float64))
                prev_ob = np.asarray(ob, np.float32)
                ob, _, terminated, truncated, info = env.step(np.asarray(a, np.float32).copy())
                done = terminated or truncated
                observations.append(prev_ob)
                actions.append(np.asarray(a, np.float32).copy())
                next_observations.append(np.asarray(ob, np.float32))
                successes.append(bool(info.get('success', False)))
        episode_ends.append(len(actions))

    episode_ends_arr = np.asarray(episode_ends, np.int32)
    successes_arr = np.asarray(successes, np.bool_)
    success_rate = float(successes_arr[episode_ends_arr - 1].mean()) if len(episode_ends_arr) else 0.0

    rollout = dict(
        observations=np.asarray(observations, np.float32),
        actions=np.asarray(actions, np.float32),
        next_observations=np.asarray(next_observations, np.float32),
        successes=successes_arr,
        episode_ends=episode_ends_arr,
        chunk_size=np.array(chunk_size),
    )
    if record_sim_state:
        rollout['qpos'] = np.asarray(qpos_list, np.float64)
        rollout['qvel'] = np.asarray(qvel_list, np.float64)
    return rollout, success_rate


def label_rollout_with_oracle(rollout: dict, goal_xyz, chunk_size: int, num_workers: int, max_oracle_steps: int, warmup_steps: int):
    """Oracle-label a freshly collected rollout's chunk boundaries; return progress samples."""
    from awr.annotate_ogbench import annotate_states
    from awr.classifier.ogbench_dataset import build_chunk_progress
    from awr.classifier.ogbench_dataset import episode_ends as chunk_episode_ends

    observations = rollout['observations']
    actions = rollout['actions']
    n = len(observations)
    terminals = np.zeros(n, np.float32)
    terminals[np.asarray(rollout['episode_ends'], np.int64) - 1] = 1.0

    needed = []
    start = 0
    for end in chunk_episode_ends(terminals, n):
        needed.append(start + np.arange(0, end - start, chunk_size))
        start = end
    needed = np.unique(np.concatenate(needed)) if needed else np.zeros(0, np.int64)

    distance = np.full(n, -1, np.int32)
    if len(needed) > 0:
        distance[needed] = annotate_states(
            rollout['qpos'], rollout['qvel'], goal_xyz, needed, num_workers, max_oracle_steps, warmup_steps,
        )

    return build_chunk_progress(observations, actions, terminals, distance, chunk_size)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env_name', default='cube-single-play-singletask-task1-v0')
    p.add_argument('--advantage', choices=['iql', 'classifier'], default='iql')
    p.add_argument('--annotated_path', default=None, help='(--advantage classifier) Oracle-annotated npz for the initial buffer')
    p.add_argument('--chunk_size', type=int, default=1)

    p.add_argument('--warm_start_steps', type=int, default=20_000, help='Initial joint fit steps before the online rounds')
    p.add_argument('--max_rounds', type=int, default=20)
    p.add_argument('--episodes_per_round', type=int, default=100)
    p.add_argument('--iql_steps_per_round', type=int, default=5000, help='Joint update steps/round, phase 1 ("train IQL"/"train classifier")')
    p.add_argument('--policy_steps_per_round', type=int, default=5000, help='Joint update steps/round, phase 2 ("train policy")')
    p.add_argument('--collect_temperature', type=float, default=1.0)

    p.add_argument('--eval_episodes', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=10.0)
    p.add_argument('--expectile', type=float, default=0.9)
    p.add_argument('--discount', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--classifier_hidden', type=int, default=256)
    p.add_argument('--val_ratio', type=float, default=0.1, help='(--advantage classifier) Val split for classifier')
    p.add_argument(
        '--classifier_update_every',
        type=int,
        default=10,
        help='(--advantage classifier) Update classifier every N actor steps',
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--data_percent', type=float, default=100.0, help='Percent of OGBench play transitions to seed the initial buffer with')
    p.add_argument('--log_interval', type=int, default=1000)

    p.add_argument('--num_workers', type=int, default=10, help='(--advantage classifier) Parallel oracle-labeling workers')
    p.add_argument('--max_oracle_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=2)

    p.add_argument('--device', choices=['cpu', 'auto'], default='cpu')
    p.add_argument('--dataset_dir', default=None)
    p.add_argument('--wandb_project', default='awr-online-ogbench')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()

    if not (0 < args.data_percent <= 100):
        p.error(f'--data_percent must be in (0, 100], got {args.data_percent}')
    if args.chunk_size < 1:
        p.error(f'--chunk_size must be >= 1, got {args.chunk_size}')
    if args.advantage == 'classifier' and not args.annotated_path:
        p.error('--advantage classifier requires --annotated_path (see awr/annotate_ogbench.py)')

    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import ogbench
    import wandb

    from awr.offline import evaluate_iql

    wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode, config=vars(args))

    kwargs = {}
    if args.dataset_dir is not None:
        kwargs['dataset_dir'] = args.dataset_dir
    print(f'loading OGBench dataset: {args.env_name}  chunk_size={args.chunk_size}')
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(args.env_name, **kwargs)

    if args.advantage == 'classifier':
        from awr.annotate_ogbench import parse_task_id, task_goal_xyz
        from awr.classifier.ogbench_dataset import ClassifierOgbenchDataset
        from awr.classifier.train_awr import train_classifier_awr

        goal_xyz = task_goal_xyz(parse_task_id(args.env_name))

        print(f'loading oracle-annotated data: {args.annotated_path}')
        annotated = dict(np.load(args.annotated_path))
        dataset = ClassifierOgbenchDataset.from_annotated(annotated, chunk_size=args.chunk_size)
        dataset = dataset.subsample(args.data_percent, seed=args.seed)

        def fit(agent, steps, step_offset, **_unused):
            return train_classifier_awr(
                steps, dataset=dataset, agent=agent, seed=args.seed, batch_size=args.batch_size,
                alpha=args.alpha, lr=args.lr, classifier_hidden=args.classifier_hidden,
                val_ratio=args.val_ratio, classifier_update_every=args.classifier_update_every,
                log_interval=args.log_interval, wandb_run=wandb, step_offset=step_offset,
            )
    else:
        from awr.iql.dataset import IQLDataset
        from awr.iql.train import train_iql

        dataset = IQLDataset.from_ogbench(train_dataset, chunk_size=args.chunk_size)
        dataset = dataset.subsample(args.data_percent, seed=args.seed)

        def fit(agent, steps, step_offset, **_unused):
            return train_iql(
                steps, dataset=dataset, agent=agent, seed=args.seed, batch_size=args.batch_size,
                expectile=args.expectile, alpha=args.alpha, lr=args.lr, discount=args.discount, tau=args.tau,
                log_interval=args.log_interval, wandb_run=wandb, step_offset=step_offset,
            )

    print(f'initial buffer size={dataset.size}')
    wandb.log({'dataset/size': dataset.size, 'round': 0}, step=0)

    def eval_and_log(agent, step, round_idx):
        sr = evaluate_iql(agent, env, args.eval_episodes, seed=args.seed, chunk_size=args.chunk_size)
        wandb.log({'evaluation/success_rate': sr, 'round': round_idx, 'dataset/size': dataset.size}, step=step)
        print(f'round {round_idx} step {step}: eval success_rate={sr:.3f}  buffer={dataset.size}', flush=True)
        return sr

    global_step = 0
    print(f'warm start: {args.warm_start_steps} joint steps on {dataset.size} transitions')
    agent, _ = fit(None, args.warm_start_steps, global_step)
    global_step += args.warm_start_steps
    eval_and_log(agent, global_step, round_idx=0)

    for r in range(1, args.max_rounds + 1):
        rollout, collect_sr = collect_episodes(
            agent, env, args.episodes_per_round, args.chunk_size, seed=args.seed + r,
            temperature=args.collect_temperature, record_sim_state=(args.advantage == 'classifier'),
        )
        print(f'round {r}: collected {args.episodes_per_round} episodes, success_rate={collect_sr:.3f}')

        if args.advantage == 'classifier':
            obs, act, labels = label_rollout_with_oracle(
                rollout, goal_xyz, args.chunk_size, args.num_workers, args.max_oracle_steps, args.warmup_steps,
            )
            dataset.add(obs, act, labels)
        else:
            from awr.iql.dataset import transitions_from_data

            dataset.add(transitions_from_data(rollout))

        wandb.log({'collect/success_rate': collect_sr, 'dataset/size': dataset.size, 'round': r}, step=global_step)
        print(f'round {r}: buffer grown to {dataset.size}')

        # Phase 1 ("train IQL"/"train classifier") and phase 2 ("train policy") are the same
        # joint update, called twice in a row on the same (now larger) buffer.
        agent, _ = fit(agent, args.iql_steps_per_round, global_step)
        global_step += args.iql_steps_per_round
        agent, _ = fit(agent, args.policy_steps_per_round, global_step)
        global_step += args.policy_steps_per_round

        eval_and_log(agent, global_step, round_idx=r)

    wandb.finish()


if __name__ == '__main__':
    main()
