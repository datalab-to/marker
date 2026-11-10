from pathlib import Path
import shutil


SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"


def get_skill_markdown() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def install_skill(target_dir: str | Path | None = None) -> Path:
    dest_root = (
        Path(target_dir).expanduser().resolve()
        if target_dir
        else Path.cwd() / ".agents" / "skills"
    )
    if dest_root.name != "marker":
        dest_root = dest_root / "marker"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_file = dest_root / "SKILL.md"
    shutil.copyfile(SKILL_PATH, dest_file)
    return dest_file
