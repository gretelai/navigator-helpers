from navigator_helpers import NL2CodePipeline
from navigator_helpers.llms.base import init_llms


def run_pipeline(llm_config_path: str, config_path: str, num_samples: int = 10):
    llm_registry = init_llms(llm_config_path)

    pipe = NL2CodePipeline(llm_registry, config_path)
    results = pipe.run(num_samples=num_samples)

    results.display_sample()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-l", "--llm-config", type=str, required=True)
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=10)
    args = parser.parse_args()
    run_pipeline(args.llm_config, args.config, args.num_samples)
