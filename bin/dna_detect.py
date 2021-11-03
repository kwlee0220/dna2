
from dna import Box, color, DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.camera import Camera
from dna.det import DetectorLoader, ObjectDetectingProcessor


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("node", metavar='node_id', help="target node_id")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--output", metavar="file",
                        help="output detection file.", required=False)
    parser.add_argument("--output_video", metavar="file",
                        help="output video file", required=False)
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE, args.node)
    camera_info = Camera.from_conf(conf.camera)
    cap = camera_info.get_capture(sync=args.sync, begin_frame=args.begin_frame, end_frame=args.end_frame)
    
    detector = DetectorLoader.load(args.detector)
    win_name = camera_info.camera_id if args.show else None
    with ObjectDetectingProcessor(cap, detector, window_name=win_name,
                                    output_video=args.output_video, output=args.output,
                                    show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}/s" )