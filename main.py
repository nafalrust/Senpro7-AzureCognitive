import argparse
import sys
from pathlib import Path

import config
import pipeline
import storage
import visualize
import vision_service


def print_report(result) -> None:
    print("\n" + "=" * 60)
    print(f"File    : {Path(result.image_path).name} ({result.width}x{result.height})")
    print(f"Caption : {result.caption} ({result.caption_confidence:.0%})")
    print(f"Tags    : {', '.join(result.tags[:8])}")
    print("Objek   :")
    if not result.objects:
        print("  (tidak ada objek terdeteksi)")
    for obj in result.objects:
        print(
            f"  - {obj.name:<15} {obj.confidence:.0%}  "
            f"box=(x={obj.x}, y={obj.y}, w={obj.width}, h={obj.height})"
        )
    print("=" * 60 + "\n")


def run_single(args) -> None:
    config.validate()
    client = vision_service.create_client()
    image_path = Path(args.image)

    results = pipeline.analyze_images(client, [image_path], use_cache=not args.no_cache)
    if not results:
        return

    print_report(results[0])
    run_dir = config.new_run_dir()
    visualize.show_single(
        results[0],
        save_path=run_dir / f"{image_path.stem}_annotated.png",
        min_confidence=args.min_conf,
    )


def run_batch(args) -> None:
    config.validate()
    client = vision_service.create_client()

    image_paths = pipeline.list_images(Path(args.folder))[: args.limit]
    if not image_paths:
        raise SystemExit(f"Tidak ada gambar di folder: {args.folder}")

    results = pipeline.analyze_images(client, image_paths, use_cache=not args.no_cache)
    if not results:
        return

    for result in results:
        print_report(result)

    run_dir = config.new_run_dir()
    storage.save_summary_csv(results, run_dir / "summary.csv", args.min_conf)
    for result in results:
        visualize.show_single(
            result,
            save_path=run_dir / f"{Path(result.image_path).stem}_annotated.png",
            min_confidence=args.min_conf,
            show=False,
        )
    visualize.show_object_summary(
        results,
        save_path=run_dir / "object_summary.png",
        min_confidence=args.min_conf,
        show=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Azure AI Vision: caption & deteksi objek")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--min-conf", type=float, default=config.MIN_OBJECT_CONFIDENCE,
                        help="confidence minimum objek (default: %(default)s)")
    common.add_argument("--no-cache", action="store_true",
                        help="abaikan cache, panggil Azure ulang")

    single = subparsers.add_parser("single", parents=[common], help="satu gambar")
    single.add_argument("image", help="path ke file gambar")
    single.set_defaults(func=run_single)

    batch = subparsers.add_parser("batch", parents=[common], help="banyak gambar, satu file per gambar")
    batch.add_argument("--folder", default=str(config.IMAGES_DIR), help="folder gambar")
    batch.add_argument("--limit", type=int, default=None, help="jumlah gambar (default: semua)")
    batch.set_defaults(func=run_batch)

    return parser


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"single", "batch", "-h", "--help"}:
        argv.insert(0, "batch")
    arguments = build_parser().parse_args(argv)
    arguments.func(arguments)
