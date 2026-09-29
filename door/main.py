import argparse
import os

if __package__:
    from .camera import frames, video_frames
    from .client import AttendanceApiClient
else:
    from camera import frames, video_frames
    from client import AttendanceApiClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Door camera attendance client")
    parser.add_argument('--api-url', default=os.getenv('BACKEND_URL', 'http://localhost:8000'))
    parser.add_argument('--token', default=os.getenv('API_TOKEN', 'change-me'))
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--door-id', type=int, default=None)
    parser.add_argument('--sample-fps', type=float, default=5.0)
    parser.add_argument('--source-id', default='door-camera')
    parser.add_argument('--video', help='Replay an MP4 instead of opening a physical camera')
    parser.add_argument('--loop', action='store_true', help='Loop the video replay')
    parser.add_argument('--demo-replay', action='store_true', help='Use the explicitly enabled replay-demo endpoints; never use for production attendance')
    args = parser.parse_args()
    if args.demo_replay and not args.video:
        parser.error('--demo-replay requires --video')
    client = AttendanceApiClient(args.api_url, args.token, source_id=args.source_id)
    source = video_frames(args.video, args.sample_fps, args.loop) if args.video else frames(args.camera_index, args.sample_fps)
    frames_seen = matches = marks = 0
    try:
        for image in source:
            frames_seen += 1
            result = client.identify(image, replay_demo=args.demo_replay)
            recognized = result.get('matches') or ([result] if result.get('matched') else [])
            for match in recognized:
                matches += 1
                marked = client.mark(match, args.door_id, replay_demo=args.demo_replay)
                marks += int(marked.get('marked', False))
                print(f"recognized person_id={match['person_id']} similarity={match['similarity']:.3f} marked={marked['marked']}", flush=True)
    except KeyboardInterrupt:
        print("\nDemo stopped.", flush=True)
    finally:
        print(f"frames={frames_seen} matches={matches} new_attendance_rows={marks}", flush=True)


if __name__ == '__main__':
    main()
