from navigator_helpers import NL2CodePipeline


def run_pipeline(config_path: str, num_samples: int = 10):
    pipe = NL2CodePipeline(config_path)
    pipe.run(num_samples=num_samples)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=10)
    args = parser.parse_args()
    run_pipeline(args.config, args.num_samples)
