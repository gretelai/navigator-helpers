from navigator_helpers import NL2CodePipeline


def run_pipeline(config_path: str, num_samples: int = 10, llm_config_path: str = None):
    pipe = NL2CodePipeline(config_path, llm_config=llm_config_path)
    results = pipe.run(num_samples=num_samples)

    results.display_sample()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=10)
    parser.add_argument("-l", "--llm-config", type=str, default=None)
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        num_samples=args.num_samples,
        llm_config_path=args.llm_config,
    )
