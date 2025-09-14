import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--data', required=True)
    ap.add_argument('--project', default='yolov8_runs')
    ap.add_argument('--name', required=True)
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--patience', type=int, default=100)
    ap.add_argument('--resume', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.project) / args.name
    retries = 0
    max_retries = 3
    backoff = 10

    while True:
        try:
            weights = args.weights
            # If there is an existing run with weights/last.pt and --resume, use it
            last = run_dir / 'weights' / 'last.pt'
            if args.resume and last.exists():
                weights = str(last)

            model = YOLO(weights)
            model.train(
                data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=args.name,
                pretrained=True,
                exist_ok=True,
                workers=args.workers,
                patience=args.patience,
                resume=args.resume,
            )
            print('Training finished successfully.')
            break
        except KeyboardInterrupt:
            print('KeyboardInterrupt received, stopping.')
            break
        except Exception as e:
            retries += 1
            print(f'Training error: {e!r}')
            if retries > max_retries:
                print('Max retries exceeded. Exiting.')
                raise
            print(f'Retrying in {backoff}s (attempt {retries}/{max_retries})...')
            time.sleep(backoff)


if __name__ == '__main__':
    main()
