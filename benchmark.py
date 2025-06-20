import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(
        description="Run drug repositioning benchmarks for specified models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model",
        choices=['HNNDTA', 'DRML-Ensemble'],
        help="The model to benchmark."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input file."
    )

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(project_root, args.input_file)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        sys.exit(1)

    if args.model == 'HNNDTA':
        hnndta_path = os.path.join(project_root, 'HNNDTA')
        if hnndta_path not in sys.path:
            sys.path.append(hnndta_path)
        
        from benchmark_hnndta import run_hnndta_benchmark

        output_dir = os.path.join(project_root, 'data_output', 'HNNDTA')

        print("Starting HNNDTA benchmark...")
        start_time = time.perf_counter()
        run_hnndta_benchmark(input_path, output_dir)
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"HNNDTA benchmark completed in {duration:.2f} seconds.")

    elif args.model == 'DRML-Ensemble':
        print("DRML-Ensemble benchmark is not implemented.")
        pass

if __name__ == "__main__":
    main()