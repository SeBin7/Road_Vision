# main.py
import argparse
import traceback
from config import get_config
from pipeline import RoadVisionPipeline

def main():
    """애플리케이션의 메인 진입점 함수"""
    parser = argparse.ArgumentParser(
        description="Run Road Vision Pipeline with a server-side CNN and local GRU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--server_url", help="Override the feature extraction server URL.")
    parser.add_argument("--weights", help="Override the path to the model weights file.")
    
    args = parser.parse_args()

    config = get_config()

    if args.server_url:
        config['SERVER_URL'] = args.server_url
    if args.weights:
        config['CLS_WEIGHT'] = args.weights
    
    pipeline = None
    try:
        pipeline = RoadVisionPipeline(config)
        pipeline.run(args.video)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception:
        print("An unexpected error occurred. See details below:")
        traceback.print_exc()
    finally:
        # 비정상 종료 시에도 리소스 정리를 시도
        if pipeline and pipeline.cap and not pipeline.cap.isOpened():
             print("Attempting to clean up resources...")
             pipeline._cleanup()

if __name__ == "__main__":
    main()