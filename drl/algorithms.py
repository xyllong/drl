import d3rlpy
from torch.optim.lr_scheduler import CosineAnnealingLR

def BCQ(args, env, dataset, test_episodes):
    vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
    rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])
    
    beta = 0.05
    bcq = d3rlpy.algos.BCQ(actor_encoder_factory=rl_encoder,
                           actor_learning_rate=1e-3,
                           critic_encoder_factory=rl_encoder,
                           critic_learning_rate=1e-3,
                           imitator_encoder_factory=vae_encoder,
                           imitator_learning_rate=1e-3,
                           batch_size=args.batch_size,
                           lam=0.75,
                           action_flexibility=0.05,
                           n_action_samples=100,
                           use_gpu=args.gpu,
                           beta=beta,)
    
    bcq.fit(dataset,
            n_steps=args.training_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=10,
            eval_episodes=test_episodes,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                # 'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"BCQ_beta{str(beta).replace('.', '')}_{args.data_type}{'_weighted' if args.weighted else ''}_{args.dataset}_{args.seed}_trim{str(args.trim_pct).replace('.', '')}_bs{args.batch_size}")
            # experiment_name=f"DEBUG_beta{str(beta).replace('.', '')}_BCQ_{args.dataset}_{args.seed}_bs{args.batch_size}")

def CQL(args, env, dataset, test_episodes):
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQL(actor_learning_rate=1e-4,
                           critic_learning_rate=3e-4,
                           temp_learning_rate=1e-4,
                           actor_encoder_factory=encoder,
                           critic_encoder_factory=encoder,
                           batch_size=args.batch_size,
                           n_action_samples=10,
                           alpha_learning_rate=0.0,
                           conservative_weight=conservative_weight,
                           use_gpu=args.gpu)

    cql.fit(dataset,
            n_steps=args.training_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=10,
            eval_episodes=test_episodes,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                # 'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"CQL_{args.data_type}{'_weighted' if args.weighted else ''}_{args.dataset}_{args.seed}_trim{str(args.trim_pct).replace('.', '')}_bs{args.batch_size}")

def IQL(args, env, dataset, test_episodes):
    # reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
    #     multiplier=1000.0)
    
    ### set maxmin explicitly for halfcheetah(for now)
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        return_max=4828.45, 
        return_min=-420.39, 
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQL(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=args.batch_size,
                           weight_temp=3.0,
                           max_weight=100.0,
                           expectile=0.7,
                           reward_scaler=reward_scaler,
                           use_gpu=args.gpu)

    # workaround for learning scheduler
    iql.create_impl(dataset[0].get_observation_shape(), dataset[0].get_action_size())
    scheduler = CosineAnnealingLR(iql.impl._actor_optim, args.training_steps)

    def callback(algo, epoch, total_step):
        scheduler.step()

    iql.fit(dataset,
            n_steps=args.training_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            save_interval=10,
            callback=callback,
            eval_episodes=test_episodes,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                # 'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"IQL_{args.data_type}{'_weighted' if args.weighted else ''}_{args.dataset}_{args.seed}_trim{str(args.trim_pct).replace('.', '')}_bs{args.batch_size}")
