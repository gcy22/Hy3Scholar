from __future__ import annotations

import json
from pathlib import Path
import subprocess
import wave

import imageio_ffmpeg


DEMO_DIR = Path(__file__).resolve().parent
BUILD_DIR = DEMO_DIR / "build"
SLIDES_DIR = BUILD_DIR / "slides"
AUDIO_DIR = BUILD_DIR / "audio"
CLIPS_DIR = BUILD_DIR / "clips"
OUTPUT = DEMO_DIR / "Hy3Scholar_功能演示_1080p.mp4"
SRT_OUTPUT = DEMO_DIR / "Hy3Scholar_功能演示_中文字幕.srt"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = json.loads((DEMO_DIR / "narration.json").read_text(encoding="utf-8"))
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    subtitles: list[str] = []
    cursor = 0.0

    for segment in segments:
        index = int(segment["id"])
        slide = SLIDES_DIR / f"{index:02d}.png"
        audio = AUDIO_DIR / f"{index:02d}.wav"
        if not slide.exists() or not audio.exists():
            raise FileNotFoundError(f"缺少 Demo 素材：{slide} 或 {audio}")
        speech_duration = wav_duration(audio)
        duration = speech_duration + 0.35
        durations.append(duration)
        clip = CLIPS_DIR / f"{index:02d}.mp4"
        fade_out = max(0.0, duration - 0.32)
        run([
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-framerate", "30", "-i", str(slide),
            "-i", str(audio),
            "-vf", f"scale=1920:1080,fade=t=in:st=0:d=0.28,fade=t=out:st={fade_out:.3f}:d=0.28",
            "-af", "apad=pad_dur=0.35",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "30",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(clip),
        ])
        start = cursor
        end = cursor + speech_duration
        subtitles.extend([
            str(index),
            f"{timestamp(start)} --> {timestamp(end)}",
            segment["narration"],
            "",
        ])
        cursor += duration
        print(f"Clip {index:02d}: {duration:.2f}s")

    total_duration = sum(durations)
    if total_duration >= 120:
        raise RuntimeError(f"视频预计 {total_duration:.2f} 秒，超过 2 分钟限制")

    concat_path = BUILD_DIR / "concat.txt"
    concat_path.write_text(
        "\n".join(f"file '{(CLIPS_DIR / f'{index:02d}.mp4').as_posix()}'" for index in range(1, len(segments) + 1)),
        encoding="utf-8",
    )
    run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-c", "copy", "-movflags", "+faststart", str(OUTPUT),
    ])
    SRT_OUTPUT.write_text("\n".join(subtitles), encoding="utf-8-sig")
    print(json.dumps({
        "output": str(OUTPUT),
        "duration_seconds": round(total_duration, 2),
        "resolution": "1920x1080",
        "fps": 30,
        "subtitles": str(SRT_OUTPUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
