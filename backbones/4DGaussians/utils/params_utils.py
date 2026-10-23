def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if param == "ModelParams":
                    setattr(args, key, value)
                elif hasattr(args, key):
                    setattr(args, key, value)

    return args