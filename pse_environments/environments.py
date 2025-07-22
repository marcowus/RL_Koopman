# NOTE: If you load, deepcopy, or assign a model/ensemble, always call move_model_to_device(model, device) on both the ensemble and each submodel.
if __name__ == "__main__":
    import env_classes
else:
    try: # TODO: find a real solution for this
        from . import env_classes
    except:
        import env_classes

def get_environment(process, objective_type,
                    process_kwargs={"delta_t" : {'timestep':15*60.0, 'control':60*60.0}}, env_kwargs={}, device=None):
    if process == 'CSTR1':
        try:
            import cstr
        except:
            from . import cstr
        process_model = cstr.CSTR1(**process_kwargs, device=device)  # Pass device
    else:
        raise NotImplementedError('Process not implemented!')

    if objective_type == 'DemandResponse':
        env = env_classes.DemandResponseEnvironment(process_model, **env_kwargs, device=device)  # Pass device
    else:
        raise NotImplementedError('Objective type not implemented!')

    return env

# Testing
if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np

    env = get_environment(process = 'CSTR1', objective_type = 'DemandResponse')

    n_episodes = 0
    env.reset()
    for i_step in tqdm(range(100)):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = \
            env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            n_episodes += 1

    print(f"n_episodes: {n_episodes}")
    print('Done!')